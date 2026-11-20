import argparse

def get_args():
    parser = argparse.ArgumentParser("DDI few-shot training")
    parser.add_argument("--gcn_mode", default="hypergin", choices=["hypergin", "gin"])
    parser.add_argument("--pair_mode", default="dual_hg", choices=["dual_hg", "plain"])
    parser.add_argument("--view_ablation", default="full", choices=["full", "wo_str", "wo_proto"])
    parser.add_argument("--metric_mode", default="relation", choices=["relation"])
    parser.add_argument("--hyp_c", type=float, default=5)         # 1.0
    parser.add_argument("--beta_str", type=float, default=0.04)
    parser.add_argument("--beta_proto", type=float, default=0.0075)
    parser.add_argument("--num_hypernodes", type=int, default=2)  # 8
    parser.add_argument("--fold", type=int, default=4, choices=[0, 1, 2, 3, 4])

    parser.add_argument("--ckpt_root", default="models1005")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()
