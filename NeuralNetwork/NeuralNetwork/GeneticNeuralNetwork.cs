using System;
using System.Collections.Generic;
using System.Text;

namespace NN
{
    public class GeneticNeuralNetwork : NeuralNetwork, IComparable<GeneticNeuralNetwork>
    {
        public double Fitness { get; set; }

        public GeneticNeuralNetwork(int nbInputs, params int[] nbNeuronPerLayer) : base(nbInputs, nbNeuronPerLayer) {}
        public GeneticNeuralNetwork(int nbInputs, double minWeight, double maxWeight, params int[] nbNeuronPerLayer) : base(nbInputs,minWeight,maxWeight,nbNeuronPerLayer) {}

        public static GeneticNeuralNetwork Reproduce(GeneticNeuralNetwork a, GeneticNeuralNetwork b, double crossingPoint = 0.5)
        {
            if (a.Layers.Length != b.Layers.Length)
                throw new Exception("Impossible de reproduire deux réseaux n'ayant pas la même topologie (" + a.Layers.Length + " couches != " + b.Layers.Length + " couches");

            // On récupère le nombre d'entrée par neurone chaque couche et on vérifie au passage la comptabilité des deux réseaux (peu prendre du temps sur de gros réseaux)
            int[] neuronsPerLayer = new int[a.Layers.Length];
            for (int li = 0; li < a.Layers.Length; li++)
            {
                if (a[li].Length != b[li].Length)
                    throw new Exception("Impossible de reproduire deux réseaux n'ayant pas la même topologie (" + a[li].Length + " neurones sur la couche " + li + " != " + b[li].Length + " neurones sur la même couche");

                neuronsPerLayer[li] = a[li].Length;

                for (int ni = 0; ni < a[li].Length; ni++)
                {
                    if (a[li][ni].Inputs.Length != b[li][ni].Inputs.Length)
                        throw new Exception("Impossible de reproduire deux réseau n'ayant pas la même topologie (" + a[li][ni].Inputs.Length + " entrées sur les neurones de la couche " + li + " != " + b[li][ni].Inputs.Length + " entrées sur les neurones de la même couche");
                }

            }

            GeneticNeuralNetwork child = new GeneticNeuralNetwork(a[0][0].Inputs.Length, neuronsPerLayer);

            // On effectue le croisement des gènes = poids
            for (int li = 0; li < child.Layers.Length; li++)
            {
                for (int ni = 0; ni < child[li].Length; ni++)
                {
                    for (int wi = 0; wi < child[li][ni].Weights.Length; wi++)
                    {
                        if (Neuron.RandomSingleton.NextDouble() >= crossingPoint)
                            child[li][ni][wi] = a[li][ni][wi];
                        else
                            child[li][ni][wi] = b[li][ni][wi];
                    }
                }
            }

            return child;
        }

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
