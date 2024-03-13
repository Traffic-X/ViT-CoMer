# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmdet.apis import init_detector
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
from mmdet.core import get_classes
import mmcv
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.
    Returns:
        (list[Tensor]): The detection result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # forward the model
    with torch.no_grad():
        result = model.backbone(data['img'][0])
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img', help='Image file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # test a single image
    results = inference_detector(model, args.img)
    mmcv.mkdir_or_exist("visual/")
    for scale_index, result in enumerate(results):
        result = result.squeeze(0).mean(0).unsqueeze(0) # calculate mean
        print(result.shape)
        for channel_index in range(result.size(0)):
            channel = result[channel_index]
            min_, max_ = channel.min(), channel.max()
            channel = (channel - min_) / (max_ - min_) # normalize
            channel = channel.cpu().numpy()
            plt.figure()
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
            plt.margins(0, 0)

            plt.imshow(channel, cmap='viridis')
            plt.savefig(f'visual/{scale_index}_{channel_index}.png',
                        bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    main()

