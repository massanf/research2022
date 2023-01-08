import json
import socket

from IPython.core.magic import register_line_magic
import requests


@register_line_magic
def slack(line):
    webhooks_url = 'https://hooks.slack.com/services/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'
    payload = {
        'text':
        line
        if line else 'A cell that was running on *{}* has finished.'.format(
            socket.gethostname())
    }
    r = requests.post(webhooks_url, data=json.dumps(payload))