using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    public class GeneticNeuralNetwork : NeuralNetwork, IComparable<GeneticNeuralNetwork>
    {
        public double Fitness { get; set; }

        public GeneticNeuralNetwork(int nbInputs, params int[] nbNeuronPerLayer) : base(nbInputs, nbNeuronPerLayer) {}

        public void Mutate(double genesToMutateRate, double mutationRate)
        {
            int genesToMutate = (int)Math.Ceiling(TotalWeights * genesToMutateRate);
            int wi;
            double mutatedWeight;
            for (int g = 0; g < genesToMutate; g++)
            {
                wi = RandomSingleton.Next(TotalWeights);
                mutatedWeight = GetWeight(wi) + (RandomSingleton.NextDouble() - 0.5) * mutationRate;
                SetWeight(wi, mutatedWeight);
            }
        }

        public int CompareTo(GeneticNeuralNetwork other)
        {
            if (Fitness > other.Fitness)
            {
                return 1;
            }
            else if (Fitness < other.Fitness)
            {
                return -1;
            }

            return 0;
        }
    }
}
