from pathlib import Path
import sys
import rioxarray as rioxr
import os
# sys.path.append("..")

from src.models import Model
from src.analysis import plot_results


def kenya_crop_type_mapper():
    data_dir = "data"

    test_folder = Path("data//test")
    test_files = test_folder.glob("*.tif")
    print(test_files)

    model_path = "data//lightning_logs//version_10//checkpoints//epoch=60.ckpt"
    print(f"Using model {model_path}")

    model = Model.load_from_checkpoint(model_path)

    for test_path in test_files:

        save_dir = Path(data_dir) / "Autoencoder"
        save_dir.mkdir(exist_ok=True)

        print(f"Running for {test_path}")

        savepath = save_dir / f"preds_{test_path.name}"
        if savepath.exists():
            print("File already generated. Skipping")
            continue

        # out_forecasted = model.predict(test_path, with_forecaster=True)
        # plot_results(out_forecasted, test_path, savepath=save_dir, prefix="forecasted_")

        out_normal = model.predict(test_path, with_forecaster=False)
        mask = out_normal.prediction_0.rename({'lon':'x','lat':'y'})
        mask.rio.to_raster(os.path.join(save_dir,test_path.name.split('.')[0]+'_mask.tif'))
        plot_results(out_normal, test_path, savepath=save_dir, prefix="full_input_")

        # out_forecasted.to_netcdf(save_dir / f"preds_forecasted_{test_path.name}.nc")
        # out_normal.to_netcdf(save_dir / f"preds_normal_{test_path.name}.nc")


if __name__ == "__main__":
    kenya_crop_type_mapper()