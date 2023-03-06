## 実行方法

以下のコード等でレポジトリをクローンし、`conda`の環境を構築してください。
```sh
! git clone git@github.com:trombiano1/research2022.git
! cd research2022
! conda create --name xrayproduction --file xrayproduction.txt
! conda activate xrayproduction
```

`research2022/data/ヴォリューム名/電圧/000***.img`となるように階層を作り、X線ファイル(`.dmg`)を保存してください。(例: `research2022/data/phantom1/120/00000.img`, `research2022/data/phantom1/120/00001.img`, ...)

以下のコマンドを実行してください。ヴォリューム名と電圧は置き換えてください。
```sh
! python xray2fbp.py ヴォリューム名 電圧
```
(例: `! python xray2fbp.py phantom1 120`)

次に、以下のコマンドを実行してください。
```sh
! cd pix2pix
! conda create --name pix2pix -f environment.yml
! conda activate pix2pix
! python test.py --dataroot ./datasets/ctfbp --name ctfbp_pix2pix --model pix2pix --direction BtoA
```

結果は`research2022/pix2pix/results/images`に保存されます。