from SLF import SignedLatentFactorModel
from utils import parameter_parser, args_printer, sign_prediction_printer, link_prediction_printer


def main():
    args = parameter_parser()
    args_printer(args)

    SLF = SignedLatentFactorModel(args)
    SLF.fit()
    SLF.save_emb()

    if args.sign_prediction:
        sign_prediction_printer(SLF.logs)
    if args.link_prediction:
        link_prediction_printer(SLF.logs)


if __name__ == "__main__":
    main()
