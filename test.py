"""画像間変換のための汎用テストスクリプト。

train.py でモデル学習後、このスクリプトでモデルをテストできます。
'--checkpoints_dir' から保存済みモデルを読み込み、結果を '--results_dir' に保存します。

まず、オプションに基づいてモデルとデータセットを作成し、一部パラメータを固定します。
次に '--num_test' 枚の画像で推論を行い、結果を HTML に保存します。

例（事前に学習するか、サイトから学習済みモデルをダウンロードしてください）:
    CycleGAN モデルをテスト（両方向）:
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    CycleGAN モデルをテスト（片方向のみ）:
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    オプション '--model test' は CycleGAN の片方向のみの結果生成に使います。
    このオプションは '--dataset_mode single' を自動設定し、片側の画像だけを読み込みます。
    一方、'--model cycle_gan' は両方向の読み込みと生成が必要で、不要な場合もあります。
    結果は ./results/ に保存されます。
    保存先は '--results_dir <directory_path_to_save_result>' で指定できます。

    pix2pix モデルをテスト:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

テストオプションの詳細は options/base_options.py と options/test_options.py を参照してください。
学習とテストのヒント: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
よくある質問: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch

try:
    import wandb
except ImportError:
    print('警告: wandb パッケージが見つかりません。オプション "--use_wandb" はエラーになります。')


if __name__ == "__main__":
    opt = TestOptions().parse()  # テストオプションを取得
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # テスト用に一部パラメータを固定
    opt.num_threads = 0  # テストは num_threads = 0 のみ対応
    opt.batch_size = 1  # テストは batch_size = 1 のみ対応
    opt.serial_batches = True  # データのシャッフルを無効化（ランダム画像での結果が必要ならコメントアウト）
    opt.no_flip = True  # 反転なし（反転画像の結果が必要ならコメントアウト）

    dataset = create_dataset(opt)  # opt.dataset_mode などのオプションに基づいてデータセットを作成
    model = create_model(opt)  # opt.model などのオプションに基づいてモデルを作成
    model.setup(opt)  # 通常のセットアップ: ネットワークの読み込みと表示、スケジューラの作成

    # Web ページを作成
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # Web ページの保存先ディレクトリ
    if opt.load_iter > 0:  # load_iter の既定値は 0
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    print(f"Web ディレクトリを作成中: {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")
    # eval モードでテスト。これは batchnorm と dropout などの層にのみ影響します。
    # [pix2pix]: 元の pix2pix では batchnorm と dropout を使うため、eval() あり/なしを試せます。
    # [CycleGAN]: CycleGAN は dropout なしの instancenorm を使うため、影響はほぼありません。
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # opt.num_test 枚だけ推論
            break
        model.set_input(data)  # データローダからデータを展開
        model.test()  # 推論を実行
        visuals = model.get_current_visuals()  # 画像結果を取得
        img_path = model.get_image_paths()  # 画像パスを取得
        if i % 5 == 0:  # HTML へ画像を保存
            print(f"{i:04d} 枚目を処理中... {img_path}")
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # HTML を保存
