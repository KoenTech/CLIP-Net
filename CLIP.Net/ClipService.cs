using System.IO;

namespace CLIP.Net
{
    public class ClipService
    {
        private readonly ClipImage _clipImage;
        private readonly ClipText _clipText;
        private readonly Tokenizer _tokenizer;
        public ClipService(string imageModelPath, string textModelPath, string vocabPath)
        {
            _clipImage = new ClipImage(imageModelPath);
            _clipText = new ClipText(textModelPath);
            _tokenizer = new Tokenizer(vocabPath);
        }

        public float[] GetTextEmbeddings(string text) => _clipText.GetEmbeddings(_tokenizer.EncodeForCLIP(text));

        public float[] GetImageEmbeddings(string imagePath) => _clipImage.GetEmbeddings(ImageConverter.GetImageData(imagePath));
        public float[] GetImageEmbeddingsVips(string imagePath) => _clipImage.GetEmbeddings(VipsImageConverter.GetImageData(imagePath));
        public float[] GetImageEmbeddings(Stream imageData) => _clipImage.GetEmbeddings(ImageConverter.GetImageData(imageData));

        public float Compare(float[] A, float[] B) => EmbeddingsComparer.Compare(A, B);
    }
}
