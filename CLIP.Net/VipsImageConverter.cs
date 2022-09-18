using Microsoft.ML.OnnxRuntime.Tensors;
using NetVips;
using System;
using System.IO;

namespace CLIP.Net;
public static class VipsImageConverter
{
    public static Memory<Float16> GetImageData(string imagePath)
    {
        using (var stream = File.OpenRead(imagePath))
        {
            using var image = Image.ThumbnailStream(stream, 224, null, 224, Enums.Size.Force);

            var data = image.WriteToMemory();

            if (image.Bands < 3)
            {
                Array.Resize(ref data, 224*224*3);
                Array.Clear(data, 224*224, 224*224*2);
            }

            var array = new Float16[data.Length];

            int n = 0;
            for (int i = 0; i < data.Length; i+=3)
            {
                array[n] = BitConverter.HalfToUInt16Bits((Half)(((data[i] / 255f) - 0.48145466f) / 0.26862954f));
                array[n+224*224] = BitConverter.HalfToUInt16Bits((Half)(((data[i+1] / 255f) - 0.4578275f) / 0.26130258f));
                array[n+224*224*2] = BitConverter.HalfToUInt16Bits((Half)(((data[i+2] / 255f) - 0.40821073f) / 0.27577711f));
                n++;
            }

            return array;
        }
    }
}

