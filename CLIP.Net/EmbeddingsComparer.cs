using System;

namespace CLIP.Net
{
    public static class EmbeddingsComparer
    {
        public static float Compare(float[] A, float[] B)
        {
            if (A.Length != B.Length)
            {
                throw new Exception("Unequal lengths!");
            }

            float dotProduct = 0f;
            float mA = 0f;
            float mB = 0f;

            for (int i = 0; i < A.Length; i++)
            {
                dotProduct += A[i] * B[i];
                mA += A[i] * A[i];
                mB += B[i] * B[i];
            }
            mA = (float)Math.Sqrt(mA);
            mB = (float)Math.Sqrt(mB);

            return dotProduct / (mA * mB);
        }
    }
}
