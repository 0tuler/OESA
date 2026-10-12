import os
import smtplib
import ssl
from email.message import EmailMessage
from sys import argv
from typing import List


def enviar_email(email: str, senha: str, sobre: str, pasta: str):
    """
    Retorna o relatorio em csv em email
    """
    # Informaçoes de email
    smtp_server = "smtp.gmail.com"
    smtp_port = 465  # Se for usar SSL
    remetente_email = email
    destinatario_email = email

    # Corpo da mensagem
    msg = EmailMessage()
    msg["From"] = remetente_email
    msg["To"] = destinatario_email

    msg["Subject"] = f"{sobre}"
    msg.set_content(sobre)

    arquivos = [
        file
        for file in os.listdir(path=pasta)
        if file.endswith(".png")
        or file.endswith(".csv")
        or file.endswith(".jpg")
        or file.endswith(".txt")
    ]

    for nome_arquivo in arquivos:
        if not os.path.isfile(nome_arquivo):
            print(f"ERRO: Arquivo '{nome_arquivo}' não encontrado.")
            exit(1)

        # Pega a extensão do arquivo
        extensão = nome_arquivo.split(".")[-1].lower()

        with open(nome_arquivo, "rb") as fd:
            dados_arquivo = fd.read()
            if extensão == "csv":
                msg.add_attachment(
                    dados_arquivo, maintype="text", subtype="csv", filename=nome_arquivo
                )
            elif extensão == "txt":
                msg.add_attachment(
                    dados_arquivo,
                    maintype="text",
                    subtype="plain",
                    filename=nome_arquivo,
                )
            elif extensão == "png":
                msg.add_attachment(
                    dados_arquivo,
                    maintype="image",
                    subtype="png",
                    filename=nome_arquivo,
                )
            elif extensão == "jpg":
                msg.add_attachment(
                    dados_arquivo,
                    maintype="image",
                    subtype="jpeg",
                    filename=nome_arquivo,
                )
            else:
                print(f"ERRO: Tipo de arquivo não suportado: {extensão}!")
                continue

    # Tentativa de envio
    try:
        # Contexto SSL
        context = ssl.create_default_context()

        # Conecta ao servidor e envia o email
        with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
            server.login(remetente_email, senha)
            server.send_message(msg)
        print("Email enviado com sucesso!")

    except Exception as e:
        print(f"Erro: {e}")


if __name__ == "__main__":
    if len(argv) > 4:
        enviar_email(argv[1], argv[2], argv[3], argv[4])
