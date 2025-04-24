import torch
import torchvision.transforms.v2 as T


class ImageShuffle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 1, "step": 0.05},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 75.0, "min": 0.0, "max": 1000.0, "step": 1},
                ),
                "sigma": (
                    "FLOAT",
                    {"default": 9.0, "min": 0.0, "max": 1000.0, "step": 1},
                ),
                "blur": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = "ipadapter/utils"

    def make_noise(self, strength, alpha, sigma, blur, image):
        seed = (
            int(torch.sum(image).item()) % 1000000007
        )  # hash the image to get a seed, grants predictability
        torch.manual_seed(seed)

        transforms = T.Compose(
            [
                T.ElasticTransform(alpha=alpha, sigma=(1 - strength) * sigma, fill=1),
                T.RandomVerticalFlip(p=1.0),
                T.RandomHorizontalFlip(p=1.0),
                T.CenterCrop(min(image.shape[1], image.shape[2])),
                T.Resize(
                    (512, 512),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
            ]
        )
        image = transforms(image.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        noise = image

        del image
        noise = torch.clamp(noise, 0, 1)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            noise = T.functional.gaussian_blur(
                noise.permute([0, 3, 1, 2]), blur
            ).permute([0, 2, 3, 1])

        # mean = torch.mean(noise, dim=[2, 3], keepdim=True)
        # std = torch.std(noise, dim=[2, 3], keepdim=True) + 1e-6
        # normalized = (noise - mean) / std
        # # Apply original statistics
        # noise = normalized * orig_std + orig_mean

        return (noise,)


NODE_CLASS_MAPPINGS = {
    "ImageShuffle": ImageShuffle,
}
