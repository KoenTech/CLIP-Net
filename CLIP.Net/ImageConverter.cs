using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.IO;
using System.Threading.Tasks;

namespace CLIP.Net
{
    public static class ImageConverter
    {
        public static Memory<Float16> GetImageData(Stream imageData)
        {
            using Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageData);
            image.Mutate(x => x.Resize(224, 224));
            Rgb24[] data = new Rgb24[224 * 224];
            image.CopyPixelDataTo(data);
            var array = new Float16[data.Length * 3];

            Parallel.For(0, data.Length, i =>
            {
                array[i] = BitConverter.HalfToUInt16Bits((Half)(((data[i].R / 255f) - 0.48145466f) / 0.26862954f));
                array[i+(224*224)] = BitConverter.HalfToUInt16Bits((Half)(((data[i].G / 255f) - 0.4578275f) / 0.26130258f));
                array[i+((224*224)+(224*224))] = BitConverter.HalfToUInt16Bits((Half)(((data[i].B / 255f) - 0.40821073f) / 0.27577711f));
            });

            return array;
        }

        public static Memory<Float16> GetImageData(string imagePath)
        {
            using Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);

            image.Mutate(x => x.Resize(224, 224));
            Rgb24[] data = new Rgb24[224 * 224];
            image.CopyPixelDataTo(data);
            var array = new Float16[data.Length * 3];

            Parallel.For(0, data.Length, i =>
            {
                array[i] = BitConverter.HalfToUInt16Bits((Half)(((data[i].R / 255f) - 0.48145466f) / 0.26862954f));
                array[i+(224*224)] = BitConverter.HalfToUInt16Bits((Half)(((data[i].G / 255f) - 0.4578275f) / 0.26130258f));
                array[i+((224*224)+(224*224))] = BitConverter.HalfToUInt16Bits((Half)(((data[i].B / 255f) - 0.40821073f) / 0.27577711f));
            });

            return array;
        }
    }
}
