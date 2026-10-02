import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import Config


def main():
    import torch.distributed as dist

    try:
        _main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _main():
    parser = argparse.ArgumentParser(prog="mgflow")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "config"):
        command = commands.add_parser(name)
        command.add_argument("config")
        command.add_argument("--set", action="append", default=[], metavar="KEY=TOML_VALUE")
        if name == "train":
            command.add_argument("--resume", help="checkpoint file or run directory")
    download = commands.add_parser("download")
    download.add_argument("--output", default="assets")
    download.add_argument("--post-trained", action="store_true")
    dataset = commands.add_parser("download-data")
    dataset.add_argument("--output", default="assets/T2I")
    dataset.add_argument("--images", action="store_true")
    text = commands.add_parser("encode-text")
    text.add_argument("--prompts", default="assets/T2I/metadata.jsonl")
    text.add_argument("--output", default="assets/T2I/text.npy")
    text.add_argument("--batch", type=int, default=64)
    images = commands.add_parser("encode-images")
    images.add_argument("--domain", choices=["imagenet", "t2i"], default="imagenet")
    images.add_argument("--images", required=True)
    images.add_argument("--output", required=True)
    images.add_argument("--config")
    images.add_argument("--encoders", default="assets/Encoders/encoders")
    images.add_argument("--batch", type=int, default=32)
    images.add_argument("--workers", type=int, default=4)
    fitting = commands.add_parser("fit")
    fitting.add_argument("features")
    fitting.add_argument("--components", type=int, nargs="+", choices=[1, 4, 16], required=True)
    fitting.add_argument("--encoder", choices=["SigLIP", "MAE", "Inception"], required=True)
    fitting.add_argument("--domain", choices=["imagenet", "t2i"], default="imagenet")
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--seed", type=int, default=3407)
    fitting.add_argument("--chunk", type=int, default=2048)
    fitting.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    check = commands.add_parser("check-reference")
    check.add_argument("reference")
    check.add_argument("--features")
    check.add_argument("--chunk", type=int, default=2048)
    check.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    sample = commands.add_parser("sample")
    sample.add_argument("--model", required=True)
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--count", type=int, default=50000)
    sample.add_argument("--batch", type=int, default=32)
    sample.add_argument("--seed", type=int, default=1)
    sample.add_argument("--prompts")
    metric = commands.add_parser("frechet")
    metric.add_argument("--features", required=True)
    metric.add_argument("--reference", required=True)
    metric.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    for name in ("sample-geneval", "sample-pickscore"):
        benchmark = commands.add_parser(name)
        benchmark.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
        benchmark.add_argument("--checkpoint", required=True)
        benchmark.add_argument("--prompts", required=True)
        benchmark.add_argument("--output", required=True)
        benchmark.add_argument("--seed", type=int)
        if name == "sample-geneval":
            benchmark.add_argument("--samples", type=int, default=4)
    imagenet = commands.add_parser("evaluate-imagenet")
    imagenet.add_argument("--images", required=True)
    imagenet.add_argument("--output", required=True)
    imagenet.add_argument("--stats", default="assets/Evaluation/ImageNet/stats")
    imagenet.add_argument("--encoders", default="assets/Encoders/encoders")
    imagenet.add_argument("--count", type=int, default=50000)
    imagenet.add_argument("--batch", type=int, default=64)
    imagenet.add_argument("--workers", type=int, default=4)
    imagenet.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    t2i = commands.add_parser("evaluate-t2i")
    t2i.add_argument("--output", required=True)
    for option in (
        "geneval-images",
        "geneval-repo",
        "geneval-models",
        "geneval-python",
        "geneval-config",
        "geneval-detector",
        "pickscore-images",
        "pickscore-prompts",
    ):
        t2i.add_argument("--" + option)
    t2i.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    t2i.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.command in ("train", "config"):
        config = Config.load(args.config, args.set)
        if args.command == "config":
            print(json.dumps(asdict(config), indent=2))
        else:
            from .training import train

            if args.resume:
                config.resume = args.resume
            train(config)
    elif args.command in ("download", "download-data"):
        from .assets import download, download_data

        if args.command == "download":
            download(args.output, args.post_trained)
        else:
            download_data(args.output, args.images)
    elif args.command == "encode-text":
        from .features import encode_text

        encode_text(args.prompts, args.output, batch=args.batch)
    elif args.command == "encode-images":
        from .features import encode_images

        config = Config.load(args.config) if args.config else None
        encode_images(
            config.domain if config else args.domain,
            args.images,
            args.output,
            config.encoder_weights_dir if config else args.encoders,
            args.batch,
            args.workers,
            config.text_features if config and config.joint else None,
            config.text_betas if config and config.joint else None,
        )
    elif args.command in ("fit", "check-reference"):
        from .distributed import rank, setup
        from .distributions import GaussianMixture
        from .distributions.fitting import FitOptions, check, feature_bank, fit

        device = setup(args.device)
        options = FitOptions(chunk=args.chunk)
        if args.command == "fit":
            options.seed = args.seed
            values = feature_bank(args.features)
            outputs = [Path(args.output) / f"{args.encoder}_K{k}.npz" for k in args.components]
            if any(path.exists() for path in outputs):
                parser.error("reference output already exists; choose a new directory")
            for k in args.components:
                reference = fit(values, k, args.domain, device, options)
                if rank() == 0:
                    path = Path(args.output) / f"{args.encoder}_K{k}.npz"
                    reference.save(path)
                    print(path)
        else:
            reference = GaussianMixture.load(args.reference, device)
            if reference.k > 1 and not args.features:
                parser.error("K>1 acceptance requires --features")
            result = check(
                feature_bank(args.features) if args.features else None, reference, options
            )
            if rank() == 0:
                print(json.dumps(result, indent=2))
            if result.get("converged") is False:
                raise SystemExit(1)
    elif args.command == "sample":
        from .data import prompts_from_file
        from .sampling import imagenet, text_to_image

        if args.prompts:
            text_to_image(
                args.model, args.checkpoint, prompts_from_file(args.prompts), args.output, args.seed
            )
        else:
            imagenet(args.model, args.checkpoint, args.output, args.count, args.batch, args.seed)
    elif args.command == "frechet":
        from .evaluation import frechet

        print(json.dumps({"FD": frechet(args.features, args.reference, args.device)}))
    elif args.command in ("sample-geneval", "sample-pickscore"):
        from .distributed import setup
        from .sampling import sample_benchmark

        setup()
        sample_benchmark(
            args.command.removeprefix("sample-"),
            args.model,
            args.checkpoint,
            args.prompts,
            args.output,
            getattr(args, "samples", 1),
            args.seed,
        )
    elif args.command == "evaluate-imagenet":
        from .distributed import rank, setup
        from .evaluation import evaluate_imagenet

        result = evaluate_imagenet(
            args.images,
            args.stats,
            args.encoders,
            args.output,
            args.count,
            args.batch,
            setup(args.device),
            args.workers,
        )
        if rank() == 0:
            print(json.dumps(result, allow_nan=False))
    elif args.command == "evaluate-t2i":
        from .t2i_evaluation import evaluate_t2i

        result = evaluate_t2i(
            args.output,
            geneval_images=args.geneval_images,
            geneval_repo=args.geneval_repo,
            geneval_models=args.geneval_models,
            geneval_python=args.geneval_python,
            geneval_config=args.geneval_config,
            pickscore_images=args.pickscore_images,
            pickscore_prompts=args.pickscore_prompts,
            device=args.device,
            batch=args.batch,
            geneval_detector=args.geneval_detector,
        )
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
