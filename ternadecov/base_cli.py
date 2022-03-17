"""Commandline functionality"""

import argparse
import ternadecov.cli_tools as cli


def get_argument_parser() -> argparse.ArgumentParser:
    """Return a prepared ArgumentParser"""
    parser = argparse.ArgumentParser(
        prog="ternadecov",
        description="ternadecov is a software package for  deconvolution of bulk RNA-seq "
        "samples from time series using single-cell datasets.",
    )

    # Global
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="tool")

    ######################################################
    # Deconvolution
    parser_deconvolve = subparsers.add_parser(
        "deconvolve", help="perform deconvolution"
    )
    parser_deconvolve.add_argument("--bulk-anndata", help="anndata for bulk data")
    parser_deconvolve.add_argument("--sc-anndata", help="anndata for sc data")
    parser_deconvolve.add_argument(
        "--sample-output-csv", help="output csv file for sample compositions"
    )
    parser_deconvolve.add_argument(
        "--iterations", help="number of iterations", default=5000
    )
    parser_deconvolve.add_argument(
        "--sc-celltype-column",
        help="single cell anndata obs column that indicates cell type",
    )
    parser_deconvolve.add_argument(
        "--bulk-time-column", help="bulk anndata obs column that indicates time"
    )
    parser_deconvolve.add_argument(
        "--feature-selection-method",
        help="feature selection method",
        default="overdispersed_bulk_and_high_sc",
    )
    parser_deconvolve.add_argument(
        "--hypercluster", help="perform hyperclustering", action="store_true"
    )
    parser_deconvolve.add_argument(
        "--basis-function-type",
        help="type of basis function for deconvolution",
        default="polynomial",
    )
    parser_deconvolve.add_argument(
        "--polynomial-degree", help="degree of polynomial", default=3
    )
    parser_deconvolve.add_argument(
        "--log-frequency", help="frequency of logging during fitting", default=1000
    )

    ######################################################
    # Simulation
    parser_simulate = subparsers.add_parser("simulate", help="simulate dataset")
    parser_simulate.add_argument(
        "--bulk-anndata", help="anndata for bulk data; required for betas"
    )
    parser_simulate.add_argument("--sc-anndata", help="anndata for sc data")
    parser_simulate.add_argument("--trajectory-out", help="trajectory output")

    return parser


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.tool == "deconvolve":
        cli.do_deconvolution(args)
    elif args.tool == "simulate":
        pass
    else:
        print("Error: Unknown subcommand.")
