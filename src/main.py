import timeit
import os
import pytz
from datetime import datetime

# 设置时区为中国时区
china_tz = pytz.timezone("Asia/Shanghai")


import torch
import logging
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.utils.utils import set_random_seed, save_results
from utils.config import get_args
from utils.load_model import load_model
from utils.load_data import load_all_data
from src.train_test import test
from utils.log import setup_logger


# 定义主函数
def main(args):
    start_overall = timeit.default_timer()

    exp_name = args.exper_name
    # 存放实验相关文件
    exper_dir = os.path.join(args.exper_base_dir, exp_name, args.dataset)
    checkpoint_dir = os.path.join(exper_dir, "checkpoint")
    result_dir = os.path.join(exper_dir, "result")
    result_filename = f"{result_dir}/{args.dataset}_results.json"
    # 存放模型测试输出结果
    output_dir = os.path.join(exper_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    # 创建目录
    os.makedirs(exper_dir, exist_ok=True)
    os.makedirs(exper_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    log_file = f"{exper_dir}/{exp_name}.log"
    setup_logger(log_file)
    logging.info(f"开始实验: {exp_name}")
    logging.info(f"配置信息: {args}")

    if args.use_gpu:
        if args.device == -1:  # 使用所有可用GPU
            args.device = torch.device("cuda")
            args.num_gpus = torch.cuda.device_count()
        else:  # 使用指定GPU
            args.device = torch.device(f"cuda:{args.device}")
            args.num_gpus = 1
    else:
        args.device = torch.device("cpu")
        args.num_gpus = 0
    logging.info(f"使用设备: {args.device}, GPU数量: {args.num_gpus}")
    logging.info("加载和预处理数据...")
    dataset = PyGLinkPropPredDataset(name=args.dataset, root="DATA")
    neg_sampler = dataset.negative_sampler
    data, node_feats, edge_feats, g, df, args = load_all_data(args, dataset)

    logging.info("数据加载和预处理完成。")

    metric = dataset.eval_metric

    logging.info("开始训练...")
    for run_idx in range(args.num_run):
        logging.info(
            "-------------------------------------------------------------------------------"
        )
        args.output_dir = output_dir + f"/run_{run_idx}"
        os.makedirs(args.output_dir, exist_ok=True)

        logging.info(f">>>>> Run: {run_idx} <<<<<")
        start_run = timeit.default_timer()

        torch.manual_seed(run_idx + args.seed)
        set_random_seed(run_idx + args.seed)

        save_model_dir = checkpoint_dir
        save_model_id = f"{args.dataset}_{args.seed}_{run_idx}"

        logging.info("Train link prediction task from scratch ...")
        logging.info("加载模型...")

        model = None
        if args.istrain == 1:
            model, args, link_pred_train = load_model(args)
            if args.num_gpus > 1:
                model = torch.nn.DataParallel(model)
            model = model.to(args.device)
            # logging.info(f"模型结构: {model}")
            model.train()
            model = link_pred_train(
                model.to(args.device), args, g, df, node_feats, edge_feats
            )

            # 保存训练好的模型
            if args.num_gpus > 1:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(save_model_dir, f"{save_model_id}.pt"),
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_model_dir, f"{save_model_id}.pt"),
                )
            logging.info(
                f"模型已保存到: {os.path.join(save_model_dir, f'{save_model_id}.pt')}"
            )

        # 用于测试以及其他实验
        if model is None:
            # 加载训练好的模型
            model, args, _ = load_model(args)
            if args.num_gpus > 1:
                model = torch.nn.DataParallel(model)
            model = model.to(args.device)
            if args.num_gpus > 1:
                model.module.load_state_dict(
                    torch.load(os.path.join(save_model_dir, f"{save_model_id}.pt"))
                )
            else:
                model.load_state_dict(
                    torch.load(os.path.join(save_model_dir, f"{save_model_id}.pt"))
                )
            logging.info(
                f"模型已从 {os.path.join(save_model_dir, f'{save_model_id}.pt')} 加载。"
            )

        dataset.load_test_ns()
        start_test = timeit.default_timer()
        (
            perf_mrr_test_mean,
            perf_mrr_test_std,
            perf_list_test,
            auroc_test,
            auprc_test,
        ) = test(
            "test",
            model.to(args.device),
            args,
            metric,
            neg_sampler,
            g,
            df,
            node_feats,
            edge_feats,
        )
        logging.info(f"Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        logging.info(
            f"\tTest: {metric}: {perf_mrr_test_mean: .4f} ± {perf_mrr_test_std: .4f}"
        )
        logging.info(f"\tTest: AUROC: {auroc_test: .4f} | AUPRC: {auprc_test: .4f}")
        test_time = timeit.default_timer() - start_test
        logging.info(f"\tTest: Elapsed Time (s): {test_time: .4f}")

        save_results(
            {
                "model": args.model,
                "data": args.dataset,
                "run": run_idx,
                "seed": args.seed,
                f"test {metric}": f"{perf_mrr_test_mean: .4f} +- {perf_mrr_test_std: .4f}",
                "test auroc": f"{auroc_test: .4f}",
                "test auprc": f"{auprc_test: .4f}",
                "test_time": test_time,
            },
            result_filename,
        )

        logging.info(
            f">>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<"
        )
        logging.info(
            "-------------------------------------------------------------------------------"
        )

    logging.info(
        f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}"
    )
    logging.info("==============================================================")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(args.device) / (1024 * 1024)
        logging.info(f"🔥 Peak GPU Memory Usage: {max_mem:.2f} MB")

    logging.info("==============================================================")


if __name__ == "__main__":
    args = get_args()
    main(args)
