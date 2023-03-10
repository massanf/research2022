# 実行方法

## 環境構築1
以下のコード等でレポジトリをクローンし、`conda`の環境を構築してください。
```sh
! git clone git@github.com:trombiano1/research2022.git
! cd research2022
! conda env create --name xrayproduction --file xrayproduction.txt
! conda activate xrayproduction
```

## データの配置
`research2022/data/ヴォリューム名/電圧/(4桁の数字).img`となるように階層を作り、X線ファイル(`.img`)を保存してください。(例: `research2022/data/phantom1/120/0000.img`, `research2022/data/phantom1/120/0001.img`, ...)

## 実行ステップ1
以下のコマンドを実行してください。ヴォリューム名と電圧は置き換えてください。
```sh
! python xray2fbp.py --volname ヴォリューム名 --voltage 電圧 --num_sheets 枚数 --use_range_lower 下限 --use_range_upper 上限 --spacing 間隔
```
(例: `! python xray2fbp.py --volname phantom1 --voltage 120 --num_sheets 450 --use_range_lower 0 --use_range_upper 1024 --spacing 8`)

## 環境構築2
次に、以下のコマンドを実行してください。
```sh
! cd pix2pix
! conda env create --file environment.yml
! conda activate pytorch-CycleGAN-and-pix2pix
```

## 学習済みモデルのダウンロード
[こちら](https://drive.google.com/drive/u/0/folders/1r3whStdwbfe_p_WKfh4b6VEhCguCgLJ2)から2つのファイルをダウンロードし、`research2022/pix2pix/checkpoints/ctfbp_pix2pix/`の下に配置してください.

## 実行ステップ2
```sh
! python test.py --dataroot ./datasets/ctfbp --name ctfbp_pix2pix --model pix2pix --direction BtoA --num_test 枚数
```
枚数は`xray2fbp.py`実行時に指定した

(`use_range_upper` - `use_range_lower`) / `spacing`

以下です。
結果は`research2022/pix2pix/results/images`に保存されます。