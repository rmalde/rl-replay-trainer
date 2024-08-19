import torch
import torch.nn as nn
import torch.nn.functional as F

class SkillMask(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_dim = 111
        self.out_dim = 18

        self.register_buffer('mask', torch.zeros(self.in_dim, dtype=torch.bool))
        indices = torch.tensor([
            0,1,2, # ball pos (3)
            3,4,5, # ball vel (3)
            # 43,44,45, # player rel pos (3)
            # 46,47,48, # player rel vel (3)
            49,50,51, # player pos (3)
            # 52,53,54,55,56,57,58,59,60, # player rotmat (9)
            61,62,63, # player vel (3)
            # 64,65,66, # player ang vel (3)
            # 67,68,69, # player pyr (3)
            # 70,71,72,73, # player stats (4)
            80,81,82, # opp pos (3)
            92,93,94, # opp vel (3)
            # 105,106,107, # opp rel pos (3)
            # 108,109,110, # opp rel vel (3)
        ])
        self.mask[indices] = True

    def forward(self, x: torch.FloatTensor):
        # x: (n, 111) -> (n, 18)
        return x[:, self.mask]

"""
:3 ball pos
3:6 ball vel
43:46 player rel pos
46:49 player rel vel
49:52 player pos
52:61 player rotmat
61:64 player vel
64:67 player ang vel
67:70 player pyr
70:74 player stats
80:83 opp pos
92:95 opp vel
105:108 opp rel pos
108:111 opp rel vel
"""


"""

return torch.stack([
            self.pos_proj(x[:, :3]), # ball pos
            self.vel_proj(x[:, 3:6]), # ball vel
            self.ang_vel_proj(x[:, 6:9]), # ball ang vel
            self.pad_proj(x[:, 9:43]), # pad pos
            self.rel_pos_proj(x[:, 43:46]), # player rel pos
            self.rel_vel_proj(x[:, 46:49]), # player rel vel
            self.pos_proj(x[:, 49:52]), # player pos
            self.rotmat_proj(x[:, 52:61]), # player rotmat
            self.vel_proj(x[:, 61:64]), # player vel
            self.ang_vel_proj(x[:, 64:67]), # player ang vel
            self.pyr_proj(x[:, 67:70]), # player pyr
            self.stats_proj(x[:, 70:74]), # player stats
            self.rel_pos_proj(x[:, 74:77]), # opp rel pos
            self.rel_vel_proj(x[:, 77:80]), # opp rel vel
            self.pos_proj(x[:, 80:83]), # opp pos
            self.rotmat_proj(x[:, 83:92]), # opp rotmat
            self.vel_proj(x[:, 92:95]), # opp vel
            self.ang_vel_proj(x[:, 95:98]), # opp ang vel
            self.pyr_proj(x[:, 98:101]), # opp pyr
            self.stats_proj(x[:, 101:105]), # opp stats
            self.rel_pos_proj(x[:, 105:108]), # player_opp rel pos
            self.rel_vel_proj(x[:, 108:111]), # player_opp rel vel
        ], dim=1)
"""