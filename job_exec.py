#!/usr/bin/env python3

"""
Executor de jobs para coleta de dados estatisticos.
Suporta SLURM e execução local com MPI, OpenMP, or OpenMP offload.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

TOOLS_DIR = Path(os.getcwd()) / "Tools"


def run(
    cmd: List[str], cwd: Optional[Path] = None, capture: bool = False
) -> Tuple[int, str, str]:
    """
    Roda um comando e retorna (returncode, stdout, stderr).
    Se captura é False, stdout/stderr é enviado ao terminal.
    """
    if capture:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    else:
        proc = subprocess.run(cmd, cwd=cwd)
        return proc.returncode, "", ""


def wait_for_slurm_job(job_id: str, poll_interval: int = 5) -> None:
    """Espera o SLURM job desaparecer do squeue."""
    while True:
        ret, stdout, _ = run(["squeue", "-j", job_id, "-h"], capture=True)
        if ret != 0 or not stdout.strip():
            break
        time.sleep(poll_interval)
    time.sleep(2)  # Tempo extra para garantir sincronização


def clean_create_dir(path: Path) -> None:
    """Remove o diretorio se existe e então cria um novo."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def build_command_for_iteration(
    args: argparse.Namespace,
    nproc: int,
    min_amostras: int,
    max_amostras: int,
    is_slurm: bool,
) -> List[str]:
    """
    Constroi o comando para rodar uma iteração.
    Retorna uma lista pronta para subprocess.run.
    """
    sbatch_opts = []

    if args.exclusive:
        sbatch_opts.append("--exclusive")

    sbatch_opts.append(f"--time={args.time}")

    if is_slurm:
        # Comando sbatch base
        if args.modo == "mpi":
            script = "batch_job.sh"
            sbatch_opts = ["--parsable", "--mem", args.mem, f"--ntasks={nproc}"]
        elif args.modo == "openmp":
            script = "batch_job_openmp.sh"
            sbatch_opts = [
                "--parsable",
                "--mem",
                args.mem,
                f"--cpus-per-task={nproc}",
                f"--export=ALL,OMP_PROC_BIND={args.omp_procbind},OMP_PLACES={args.omp_places}",
            ]
        else:  # openoff
            script = "batch_job_openoff.sh"
            sbatch_opts = [
                "--parsable",
                "--mem",
                args.mem,
                "--cpus-per-task=8",
                f"--gpus={args.gpus}",
            ]

        # Permite sobrescrever o script de batch padrão via --batch_script
        if getattr(args, "batch_script", None):
            script = args.batch_script

        cmd = (
            ["sbatch"]
            + sbatch_opts
            + [
                script,
                args.exec_name,
                args.output_file,
                str(min_amostras),
                str(max_amostras),
                args.sequana,
            ]
        )

        if args.modo == "openoff":
            cmd.extend([str(args.teams), str(nproc)])

        cmd.append(args.preload if args.preload else "")
        cmd.append(args.args if args.args else "")

        return cmd
    else:
        # Comando de execução local
        if args.modo == "mpi":
            cmd = ["mpirun", f"./{args.exec_name}"] + (
                args.args.split() if args.args else []
            )
        else:
            # openmp ou openoff
            cmd = [f"./{args.exec_name}"] + (args.args.split() if args.args else [])
        return cmd


def run_iteration_local(cmd: List[str], output_file: Path, num_runs: int) -> bool:
    """
    Roda o comando localmente num_runs vezes, appending a saida para o arquivo de saida
    Retorna True no sucesso geral.
    """
    success = True
    with open(output_file, "a") as f:
        for _ in range(num_runs):
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                print(f"Command error: {proc.stderr}", file=sys.stderr)
                success = False
    return success


@dataclass
class JobConfig:
    name: str
    exec_path: Path
    header: str
    sequana: str
    total_amostras: int
    passo: int
    modo: str
    preload: str
    args_str: str
    mem: str
    info: str
    tipo: str
    nprocs: List[int]
    teams: Optional[int]
    gpus: Optional[int]
    email: str
    passwd: str
    omp_procbind: str
    omp_places: str
    extra_files: List[str]
    batch_script: str
    just_email: bool
    exclusive: bool
    time: str

    @property
    def exec_name(self) -> str:
        return self.exec_path.name

    @property
    def output_file(self) -> str:
        return f"{self.name}_resultados.csv"

    @property
    def report_file(self) -> str:
        return f"{self.name}.csv"


class JobExecutor:
    def __init__(self, config: JobConfig):
        self.config = config
        self.scratch_dir: Optional[Path] = None
        self.original_cwd = Path.cwd()

    def setup_scratch(self) -> None:
        """Cria e prepara o diretorio de scratch."""
        scratch_env = os.environ.get("SCRATCH")

        if scratch_env:
            base = Path(scratch_env)
        else:
            base = Path("./scratch")
            base.mkdir(exist_ok=True)

        self.scratch_dir = base / self.config.name
        clean_create_dir(self.scratch_dir)

        # Copy executable
        shutil.copy(self.config.exec_path, self.scratch_dir / self.config.exec_name)

        # Copy tools
        if TOOLS_DIR.exists():
            for f in TOOLS_DIR.iterdir():
                if f.is_file():
                    shutil.copy(f, self.scratch_dir / f.name)

        # Copia o script de batch customizado, se um caminho válido foi informado
        # (se for apenas um nome de arquivo já presente em Tools/, não faz nada aqui,
        # pois já foi copiado no passo de "Copy tools" acima)
        if self.config.batch_script:
            custom_script = Path(self.config.batch_script)
            if custom_script.exists():
                shutil.copy(custom_script, self.scratch_dir / custom_script.name)
                # Garante que o comando sbatch use apenas o nome do arquivo,
                # já que a execução ocorre dentro do scratch_dir
                self.config.batch_script = custom_script.name
                print(f"Script de batch customizado copiado: {custom_script.name}")
            elif not (self.scratch_dir / self.config.batch_script).exists():
                print(
                    f"Aviso: batch_script '{self.config.batch_script}' não encontrado "
                    "localmente nem em Tools/. Assumindo que já existe no scratch.",
                    file=sys.stderr,
                )

        # Copia arquivos extras solicitados (ex: dados binários de entrada ou arquivos de texto)
        if self.config.extra_files:
            print("Copiando arquivos extras para o SCRATCH...")
            for file_name in self.config.extra_files:
                src_file = self.original_cwd / file_name
                if src_file.exists():
                    shutil.copy(src_file, self.scratch_dir / file_name)
                    print(f"  -> Copiado: {file_name}")
                else:
                    print(
                        f"  -> Aviso: Arquivo extra não encontrado no diretório local: {file_name}",
                        file=sys.stderr,
                    )

        os.chdir(self.scratch_dir)

    def write_header(self) -> None:
        """Escreve o CSV header no arquivo de saida."""
        with open(self.config.output_file, "w") as f:
            f.write(self.config.header + "\n")

    def run_slurm_jobs(self) -> None:
        """Envia e monitora SLURM jobs para cada nproc."""
        for nproc in self.config.nprocs:
            min_am = 1
            max_am = self.config.passo
            while min_am <= self.config.total_amostras:
                if max_am > self.config.total_amostras:
                    max_am = self.config.total_amostras

                print(f"SLURM: N={nproc}, amostras [{min_am}, {max_am}]")
                start = time.time()

                cmd = build_command_for_iteration(
                    self._as_namespace(), nproc, min_am, max_am, is_slurm=True
                )
                ret, stdout, stderr = run(cmd, capture=True)
                if ret != 0:
                    print(f"sbatch failed: {stderr}", file=sys.stderr)
                    sys.exit(1)
                job_id = stdout.strip()
                print(f"Waiting for job {job_id}...")
                wait_for_slurm_job(job_id)

                elapsed = int(time.time() - start)
                print(f"Job finished in {elapsed}s")

                min_am += self.config.passo
                max_am += self.config.passo

    def run_local_jobs(self) -> None:
        """Roda os jobs de forma local para cada nproc."""
        for nproc in self.config.nprocs:
            min_am = 1
            max_am = self.config.passo
            while min_am <= self.config.total_amostras:
                if max_am > self.config.total_amostras:
                    max_am = self.config.total_amostras

                print(f"LOCAL: N={nproc}, amostras [{min_am}, {max_am}]")
                start = time.time()

                # Set environment variables for OpenMP/offload
                if self.config.modo == "openmp":
                    os.environ["OMP_NUM_THREADS"] = str(nproc)
                elif self.config.modo == "openoff":
                    os.environ["OMP_TARGET_OFFLOAD"] = "MANDATORY"
                    os.environ["OMP_NUM_TEAMS"] = str(self.config.teams)
                    os.environ["OMP_TEAMS_THREAD_LIMIT"] = str(nproc)

                cmd = build_command_for_iteration(
                    self._as_namespace(), nproc, min_am, max_am, is_slurm=False
                )
                num_runs = max_am - min_am + 1
                success = run_iteration_local(
                    cmd, Path(self.config.output_file), num_runs
                )
                if not success:
                    print("Execução local continha erros, continuando ainda assim...")

                elapsed = int(time.time() - start)
                print(f"Iteração finalizada {elapsed}s")

                min_am += self.config.passo
                max_am += self.config.passo

    def just_email(self) -> None:
        """
        Somente envial email com o conteudo a ser processado
        """
        shutil.copy(
            self.config.output_file, self.original_cwd / self.config.output_file
        )
        print(f"Arquivo de resultados salvo: {self.config.output_file}")

        subject = f"Relatorio sobre a execucao do programa {self.config.exec_name}"
        if self.config.info:
            subject += f", {self.config.info}"
        run(
            [
                "python",
                "enviar_email.py",
                self.config.email,
                self.config.passwd,
                subject,
                "./",
            ]
        )

    def post_process(self) -> None:
        """Faz parsing, plotagem de dados e envia por email."""

        print("Resultado de parse...")
        run(
            [
                "python",
                "parse_data.py",
                self.config.output_file,
                self.config.report_file,
            ]
        )

        print("Gerando gráficos...")
        run(["python", "graficos.py", self.config.report_file])

        print("Enviando email...")
        subject = f"Relatorio sobre a execucao do programa {self.config.exec_name}"
        if self.config.info:
            subject += f", {self.config.info}"
        run(
            [
                "python",
                "enviar_email.py",
                self.config.email,
                self.config.passwd,
                subject,
                "./",
            ]
        )

        # Copia de volta o relatorio final
        shutil.copy(
            self.config.report_file, self.original_cwd / self.config.report_file
        )

        perf_out = "perf.txt"
        if Path(perf_out).exists():
            shutil.copy(perf_out, self.original_cwd / f"{self.config.name}_perf.txt")
            print(f"Ficheiro de métricas salvo como: {self.config.name}_perf.txt")

    def _as_namespace(self) -> argparse.Namespace:
        """Convert config to argparse.Namespace for compatibility with old function."""
        ns = argparse.Namespace()
        for k, v in self.config.__dict__.items():
            setattr(ns, k, v)
        ns.exec_name = self.config.exec_name
        ns.output_file = self.config.output_file
        ns.args = self.config.args_str
        return ns

    def run(self) -> None:
        """Main."""
        try:
            self.setup_scratch()
            self.write_header()
            total_start = time.time()

            if self.config.tipo == "slurm":
                self.run_slurm_jobs()
            else:
                self.run_local_jobs()

            total_elapsed = int(time.time() - total_start)
            print(
                f"Tempo total de execução: {total_elapsed}s for {self.config.output_file}"
            )

            if self.config.just_email:
                self.just_email()
            else:
                self.post_process()

        finally:
            os.chdir(self.original_cwd)
            print("Pronto.")


# Parse de argumentos
def parse_and_validate() -> JobConfig:
    parser = argparse.ArgumentParser(
        description="Executor de jobs para coleta de dados estatísticos"
    )
    # Requerido
    parser.add_argument("-n", "--name", required=True)
    parser.add_argument("-p", "--path", required=True, type=Path)
    parser.add_argument("--header", required=True)
    parser.add_argument("--sequana", required=True)
    # Opcional
    parser.add_argument("--omp_procbind", type=str, default="close")
    parser.add_argument("--omp_places", type=str, default="cores")
    parser.add_argument("-a", "--amostras", type=int, default=10)
    parser.add_argument("--passo", type=int, default=5)
    parser.add_argument(
        "--modo", choices=["mpi", "openmp", "openoff"], default="openmp"
    )
    parser.add_argument("--preload", default="")
    parser.add_argument("--args", default="")
    parser.add_argument("--mem", default="")
    parser.add_argument("--info", default="")
    parser.add_argument("--tipo", choices=["slurm", "local"], default="slurm")
    parser.add_argument("--nprocs", default="1 2 4 8 16 24 32 48")
    parser.add_argument("--teams", type=int)
    parser.add_argument("--gpus", type=int)

    # Argumento extra_files recebe múltiplos valores para copiar
    parser.add_argument(
        "--extra_files",
        nargs="*",
        default=[],
        help="Lista de arquivos no diretório atual que devem ser copiados para a pasta de scratch",
    )
    # Permite usar um script de batch SLURM diferente do padrão
    # (batch_job.sh / batch_job_openmp.sh / batch_job_openoff.sh).
    # Pode ser um caminho local (será copiado para o scratch) ou apenas
    # um nome de arquivo já presente na pasta Tools/.
    parser.add_argument(
        "--batch_script",
        default="",
        help="Caminho ou nome de um script de batch SLURM customizado, "
        "substituindo o padrão escolhido automaticamente pelo --modo",
    )

    parser.add_argument(
        "--just_email",
        action="store_true",
        help="Executa o just_email() em vez do post_process()",
    )

    parser.add_argument(
        "--exclusive",
        action="store_true",
        help="Aponta se será usado o regime de exclusividade, ou seja, se irá ocupar a sequana inteira selecionada para o job, consome mais recursos",
    )

    # Tempo máximo especificado pelo sequana_cpu_dev, talvez pensar depois em trocar esse tempo aqui para segundos e então transformar para minutos antes de enviar ao sbatch
    parser.add_argument("--time", type=str, default="0-00:20:00")

    args = parser.parse_args()

    # Converte nprocs string para list de ints
    nprocs_list = [int(x) for x in args.nprocs.split()]

    # Validação
    if args.modo == "openoff":
        if args.teams is None:
            parser.error("openoff requer --teams")
        if args.gpus is None:
            parser.error("openoff requer --gpus")

    # Pega os env para enviar email
    email = os.environ.get("email", "")
    passwd = os.environ.get("passwd", "")
    if not email or not passwd:
        print(
            "Atenção: email/passwd não setado no env. Email pode falhar.",
            file=sys.stderr,
        )

    return JobConfig(
        name=args.name,
        exec_path=args.path,
        header=args.header,
        sequana=args.sequana,
        total_amostras=args.amostras,
        passo=args.passo,
        modo=args.modo,
        preload=args.preload,
        args_str=args.args,
        mem=args.mem,
        info=args.info,
        tipo=args.tipo,
        nprocs=nprocs_list,
        teams=args.teams,
        gpus=args.gpus,
        email=email,
        passwd=passwd,
        omp_procbind=args.omp_procbind,
        omp_places=args.omp_places,
        extra_files=args.extra_files,
        batch_script=args.batch_script,
        just_email=args.just_email,
        exclusive=args.exclusive,
        time=args.time,
    )


# Ponto de entrada


def main() -> None:
    # Instala pacotes necessários
    try:
        config = parse_and_validate()
        executor = JobExecutor(config)
        executor.run()
    except KeyboardInterrupt:
        print("Cancelando...")


if __name__ == "__main__":
    main()
