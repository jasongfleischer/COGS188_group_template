import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque,defaultdict
import torch.nn.functional as F
import numpy as np
from reward_prediction import MahjongGame

game = MahjongGame()
game.deal_tiles()
while True:
    game_over = game.play_turn()  # ゲームのターンを実行し、終了かどうかを判定
    state = game.get_game_state()  # ゲームの状態を取得
    print(state)  # 現在のゲームの状態を出力

    if game_over == True:  # ゲームが終了した場合
        print("ゲームが終了しました!")
        break  # ループを終了
