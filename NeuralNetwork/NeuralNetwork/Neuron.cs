using Functions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{ 
    public class Neuron
    {
        public static Random RandomSingleton = new Random(DateTime.Now.Millisecond);
        public double[] Inputs { get; set; }
        public double[] Weights { get; set; }
        public double Bias
        {
            get
            {
                return Weights[Weights.Length - 1];
            }
            set
            {
                Weights[Weights.Length - 1] = value;
            }
        }
        public double Output { get; private set; }
        public double this[int wi]
        {
            get
            {
                return Weights[wi];
            }
            set
            {
                Weights[wi] = value;
            }
        }

        public Neuron(int nbInputs, double minInitWeight = 0, double maxInitWeight = 0)
        {
            if (nbInputs <= 0)
                throw new Exception("Un neurone ne peut pas avoir moins d'une entrée");

            Inputs = new double[nbInputs];
            Weights = new double[nbInputs + 1];

            for (int i = 0; i < Weights.Length; i++)
            {
                if (minInitWeight == maxInitWeight)
                    Weights[i] = RandomSingleton.NextDouble();
                else
                    Weights[i] = RandomSingleton.Next(minInitWeight, maxInitWeight);
            }

            // Bias
            Bias = 1;
        }

        public void ComputeOutput(double[] inputs, IActivationFunction activationFunction)
        {
            if (inputs.Length != this.Inputs.Length)
                throw new Exception("Les données d'entrées de taille " + inputs.Length + " est différent du nombre d'entrée du neurone (" + this.Inputs.Length + ")");

            this.Inputs = inputs;

            // Dotproduct = inputs[] * weights[] 
            double dotProduct = 0;
            foreach (double input in inputs)
            {
                foreach (double weight in Weights)
                {
                    dotProduct += input * weight;
                }
            }

            // Output = ActivationFunction(dotproduct + bias)
            Output = activationFunction.F(dotProduct + Bias);
        }
    }
}
