import paramiko
from scp import SCPClient
from password_config import *


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client


ssh = createSSHClient(server_ip, server_port, username, password)
scp = SCPClient(ssh.get_transport())
