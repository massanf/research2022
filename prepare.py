import importlib
import imageio.v2 as imageio
from patient import patient
import glob
from tqdm.notebook import tqdm
import json
import socket
import requests
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6ACUZJ


def slack(line):
    webhooks_url = 'https://hooks.slack.com/services/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'
    payload = {
        'text':
        line
        if line else 'A cell that was running on *{}* has finished.'.format(
            socket.gethostname())
    }
    r = requests.post(webhooks_url, data=json.dumps(payload))


def main():
    for vol_path in tqdm(glob.glob("./data/*")):
        if "__" in vol_path:
            continue
        vol = vol_path.split("/")[-1]
        print(vol)
        # print(vol)
        p = patient.patient(name=vol)

    # imageio.mimsave(f"pics/drr.gif", p.drr.img)


try:
    main()
    slack()
except KeyboardInterrupt:
    raise
except Exception as e:
    slack(e)
    raise
