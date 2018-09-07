using NN;
using System;
using System.Collections.Generic;

namespace Genetics
{
    internal static class GeneticsHelper
    {
        public static NeuralNetwork Reproduce(NeuralNetwork a, NeuralNetwork b)
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

            NeuralNetwork child = new NeuralNetwork(a[0][0].Inputs.Length, neuronsPerLayer);

            // On effectue le croisement des gènes = poids
            for (int li = 0; li < child.Layers.Length; li++)
            {
                for (int ni = 0; ni < child[li].Length; ni++)
                {
                    for (int wi = 0; wi < child[li][ni].Weights.Length; wi++)
                    {
                        if (NeuralNetwork.RandomSingleton.NextDouble() >= 0.5)
                            child[li][ni][wi] = a[li][ni][wi];
                        else
                            child[li][ni][wi] = b[li][ni][wi];
                    }
                }
            }

            return child;
        }
    }
}
