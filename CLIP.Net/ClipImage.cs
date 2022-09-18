using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CLIP.Net
{
    public class ClipImage
    {
        private readonly string _modelPath;
        private readonly InferenceSession _inferenceSession;
        public ClipImage(string modelPath)
        {
            _modelPath = modelPath;
            _inferenceSession = new InferenceSession(_modelPath);
        }

        public float[] GetEmbeddings(Memory<Float16> imageData)
        {
            var inTensor = new DenseTensor<Float16>(memory: imageData, dimensions: new int[] { 1, 3, 224, 224 });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", inTensor) };

            var output = (Tensor<Float16>)_inferenceSession.Run(input).First().Value;
            float[] embeddings = new float[512];
            for (int i = 0; i < output.Length; i++)
            {
                embeddings[i] = (float)BitConverter.UInt16BitsToHalf(output.GetValue(i));
            }
            return embeddings;
        }
    }
}
