using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CLIP.Net
{
    public class ClipText
    {
        private readonly string _modelPath;
        private readonly InferenceSession _inferenceSession;
        public ClipText(string modelPath)
        {
            _modelPath = modelPath;
            _inferenceSession = new InferenceSession(_modelPath);
        }

        public float[] GetEmbeddings(int[] tokens)
        {
            var inTensor = new DenseTensor<int>(memory: tokens, dimensions: new int[] { 1, 77 });
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
