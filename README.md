# 実行方法

## 環境構築1
以下のコード等でレポジトリをクローンし、`conda`の環境を構築してください。
```sh
! git clone git@github.com:trombiano1/research2022.git
! cd research2022
! conda create --name xrayproduction --file xrayproduction.txt
! conda activate xrayproduction
```

## データの配置
`research2022/data/ヴォリューム名/電圧/(4桁の数字).img`となるように階層を作り、X線ファイル(`.dmg`)を保存してください。(例: `research2022/data/phantom1/120/0000.img`, `research2022/data/phantom1/120/0001.img`, ...)

## 実行ステップ1
以下のコマンドを実行してください。ヴォリューム名と電圧は置き換えてください。
```sh
! python xray2fbp.py ヴォリューム名 枚数 電圧
```
(例: `! python xray2fbp.py phantom1 450 120`)

## 環境構築2
次に、以下のコマンドを実行してください。
```sh
! cd pix2pix
! conda create --name pix2pix -f environment.yml
! conda activate pix2pix
```

## 実行ステップ2
```sh
! python test.py --dataroot ./datasets/ctfbp --name ctfbp_pix2pix --model pix2pix --direction BtoA
```

結果は`research2022/pix2pix/results/images`に保存されます。