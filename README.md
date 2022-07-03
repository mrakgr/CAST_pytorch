# Main

Fork of the original [CAST repo](https://github.com/zyxElsa/CAST_pytorch). I've reduced the memory consumption by 7x and adapted it so that it uses the CPU, this allows me to use style transfer even on large 2k images. I've also made some changed to make it easier to use. Unlike the original, the version in this repo will iterate over all the files in both the content and style folder, applying style transfer to each. It has more sensible naming of the output files, and I've also fixed the bug with width scaling which does not work in the original. Finally, I've also included the [`pngquant`](https://pngquant.org/) utility so the outputed files are automatically compressed by 2/3rds without any visible reduction in quality.

For the models, check out the original repo where the links is provided. The author of the paper has them in a Google Drive folder. They just need to be unpacked into the `checkpoints` folder.

Once that is done, put the content images into `datasets/testA` and style images into `datasets/testB`. Then run `test.py`. If everything works, they will be output into the `results` folder.

You can check out some style transfer results on my [own Twitter](https://twitter.com/Ghostlike). For example [this](https://twitter.com/Ghostlike/status/1536019692414017538) or [this](https://twitter.com/Ghostlike/status/1542177545537425408).