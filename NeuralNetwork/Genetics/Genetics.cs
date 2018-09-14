using NN;
using System;
using System.Collections.Generic;

namespace Genetics
{
    public class Genetics
    {
        public List<GeneticNeuralNetwork> Population { get; set; }

        public void Init(int popSize, int nbInputs, double minWeight, double maxWeight, params int[] nbNeuronsPerLayer)
        {
            // Génération de la population
            Population = new List<GeneticNeuralNetwork>();
            for (int i = 0; i < popSize; i++)
                Population.Add(new GeneticNeuralNetwork(nbInputs, minWeight, maxWeight, nbNeuronsPerLayer));
        }
    }
}
