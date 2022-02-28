run:
	ternadecov --cuda --verbose deconvolve --bulk-anndata "/home/nbarkas/disk1/work/deconvolution_method/datasets/ebov/load_data_python/ebov_bulk.h5ad" --sc-anndata "/home/nbarkas/disk1/work/deconvolution_method/datasets/ebov/load_data_python/ebov_sc.h5ad" --sample-output-csv sample_output.csv --sc-celltype-column Subclustering_reduced --bulk-time-column "dpi_time"

lint: FORCE
	black .

FORCE:
