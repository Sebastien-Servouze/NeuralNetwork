using System;

namespace Functions
{
    public class Sigmoid : IActivationFunction
    {
        public double F(double x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }
    }
}
