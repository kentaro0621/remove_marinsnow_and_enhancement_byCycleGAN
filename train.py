"""画像間変換のための汎用トレーニングスクリプト。

このスクリプトは、さまざまなモデル（オプション '--model'：例 pix2pix, cyclegan, colorization）と
さまざまなデータセット（オプション '--dataset_mode'：例 aligned, unaligned, single, colorization）に対応します。
データセット（'--dataroot'）、実験名（'--name'）、モデル（'--model'）を指定してください。

まず、指定されたオプションに基づいてモデル、データセット、ビジュアライザを作成します。
次に、標準的なネットワークの学習を行います。学習中は画像の可視化/保存、損失のプロットの表示/保存、モデルの保存も行います。
このスクリプトは学習の継続/再開に対応しています。'--continue_train' を使うと以前の学習を再開できます。

例:
    CycleGAN モデルを学習:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    pix2pix モデルを学習:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

学習オプションの詳細は options/base_options.py と options/train_options.py を参照してください。
学習とテストのヒント: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
よくある質問: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

"""
実行
docker run --gpus all \
-p 8097:8097 \
-v $(pwd)/dataset:/workspace/dataset \
-v $(pwd)/checkpoints:/workspace/checkpoints \
-v $(pwd)/logs:/workspace/logs \
cyclegan-train \
bash -c "bash run_train.sh 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log"

ローカルの場合
bash run_train.sh

"""

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


if __name__ == "__main__":
    opt = TrainOptions().parse()  # 学習オプションを取得
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # opt.dataset_mode などのオプションに基づいてデータセットを作成
    dataset_size = len(dataset)  # データセット内の画像枚数を取得
    print(f"学習画像の枚数 = {dataset_size}")

    model = create_model(opt)  # opt.model などのオプションに基づいてモデルを作成
    model.setup(opt)  # 通常のセットアップ: ネットワークの読み込みと表示、スケジューラの作成
    visualizer = Visualizer(opt)  # 画像とプロットを表示/保存するビジュアライザを作成
    total_iters = 0  # 学習の総イテレーション数
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # エポック全体の計測開始時刻
        iter_data_time = time.time()  # イテレーションごとのデータ読み込み計測開始時刻
        epoch_iter = 0  # 現在エポックの学習イテレーション数（各エポックで 0 にリセット）
        visualizer.reset()
        # DistributedSampler 用にエポックを設定
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # 1 エポック内の内側ループ
            iter_start_time = time.time()  # イテレーションごとの計算時間計測
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # データセットからデータを展開し前処理を適用
            model.optimize_parameters()  # 損失計算、勾配取得、重み更新

            if total_iters % opt.display_freq == 0:  # visdom で画像を表示し、HTML に保存
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # 学習損失を表示し、ログを保存
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # <save_latest_freq> ごとに最新モデルを保存
                print(f"最新モデルを保存中 (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # 各エポック終了時に学習率を更新

        if epoch % opt.save_epoch_freq == 0:  # <save_epoch_freq> エポックごとにモデルを保存
            print(f"エポック {epoch} 終了時にモデルを保存 (iters {total_iters})")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"エポック {epoch} / {opt.n_epochs + opt.n_epochs_decay} 終了\t 所要時間: {time.time() - epoch_start_time:.0f} 秒")

    cleanup_ddp()
