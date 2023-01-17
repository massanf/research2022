import paramiko
from scp import SCPClient


def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

ssh = createSSHClient("192.168.193.104", "", "mfujita", "piano38125")
scp = SCPClient(ssh.get_transport())