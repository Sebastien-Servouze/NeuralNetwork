using Functions;
using System;
using System.Collections.Generic;

namespace NN
{
    public class NeuralNetwork
    {
        public static Random RandomSingleton = new Random();
        public Neuron[][] Layers { get; set; }
        public double[] Outputs { get; private set; }
        public int TotalWeights { get; private set; }
        
        public Neuron[] this[int li]
        {
            get
            {
                return Layers[li];
            }
            set
            {
                Layers[li] = value;
            }
        }

        public NeuralNetwork(int nbInputs, params int[] nbNeuronPerLayer)
        {
            Layers = new Neuron[nbNeuronPerLayer.Length][];

            // Création de la couche d'entrée
            Layers[0] = new Neuron[nbNeuronPerLayer[0]];
            for (int ni = 0; ni < Layers[0].Length; ni++)
            {
                Layers[0][ni] = new Neuron(nbInputs);
                TotalWeights += Layers[0][ni].Weights.Length;
            }

            // Création des couches cachées / sortie
            for (int li = 1; li < Layers.Length; li++)
            {
                Layers[li] = new Neuron[nbNeuronPerLayer[li]];
                for (int ni = 0; ni < Layers[li].Length; ni++)
                {
                    Layers[li][ni] = new Neuron(Layers[li - 1].Length);
                    TotalWeights += Layers[li][ni].Weights.Length;
                }
            }
        }

        public void ComputeOutput(double[] inputs, IActivationFunction activationFunction)
        {
            if (inputs.Length != Layers[0][0].Inputs.Length)
            {
                throw new Exception("Les données d'entrées de taille " + inputs.Length + " est différent du nombre d'entrée des neurones de la première couche (" + Layers[0][0].Inputs.Length + ")");
            }

            // Alimentation de la première couche
            double[] exLayerOutput = new double[Layers[0].Length];
            for (int ni = 0; ni < Layers[0].Length; ni++)
            {
                Layers[0][ni].ComputeOutput(inputs, activationFunction);
                exLayerOutput[ni] = Layers[0][ni].Output;
            }

            // Alimentation des couches cachées / sortie
            double[] currentLayerOutput;
            for (int li = 1; li < Layers.Length; li++)
            {
                currentLayerOutput = new double[Layers[li].Length];
                for (int ni = 0; ni < Layers[li].Length; ni++)
                {
                    Layers[li][ni].ComputeOutput(exLayerOutput, activationFunction);
                    currentLayerOutput[ni] = Layers[li][ni].Output;
                }

                exLayerOutput = currentLayerOutput;
            }

            Outputs = exLayerOutput;
        }

        public double GetWeight(int weightIndex)
        {
            int currentWeight = 0;
            for (int li = 0; li < Layers.Length; li++)
            {
                for (int ni = 0; ni < Layers[li].Length; ni++)
                {
                    for (int wi = 0; wi < Layers[li][ni].Weights.Length; wi++)
                    {
                        currentWeight++;
                        if (currentWeight == weightIndex)
                        {
                            return Layers[li][ni][wi];
                        }
                    }
                }
            }

            throw new Exception("Le poids " + weightIndex + " n'existe pas dans ce réseau");
        }

        public void SetWeight(int weightIndex, double weight)
        {
            int currentWeight = 0;
            for (int li = 0; li < Layers.Length; li++)
            {
                for (int ni = 0; ni < Layers[li].Length; ni++)
                {
                    for (int wi = 0; wi < Layers[li][ni].Weights.Length; wi++)
                    {
                        currentWeight++;
                        if (currentWeight == weightIndex)
                        {
                            Layers[li][ni][wi] = weight;
                            return;
                        }
                    }
                }
            }

            throw new Exception("Le poids " + weight + " n'existe pas dans ce réseau");
        }

    }
}
