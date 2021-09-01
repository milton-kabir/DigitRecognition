//package recognition;
package com.kabir.milton;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.Scanner;

class Network implements Serializable {
    double[][][] weight;
    double[][] bias;

    Network(int[] shape) {
        weight = new double[shape.length - 1][][];
        bias = new double[shape.length - 1][];
        var r = new Random();
        for (var l = 0; l < weight.length; l++) {
            weight[l] = new double[shape[l + 1]][shape[l]];
            bias[l] = new double[shape[l + 1]];
            for (var i = 0; i < weight[l].length; i++) {
                for (var j = 0; j < weight[l][i].length; j++) {
                    weight[l][i][j] = r.nextGaussian() / Math.sqrt(weight[l][i].length);
                }
            }
        }
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    double[][] activation(double[] input) {
        var activation = new double[weight.length + 1][];
        activation[0] = input;
        for (var l = 0; l < weight.length; l++) {
            activation[l + 1] = new double[weight[l].length];
            for (var i = 0; i < weight[l].length; i++) {
                for (var j = 0; j < weight[l][i].length; j++) {
                    activation[l + 1][i] += weight[l][i][j] * activation[l][j];
                }
                activation[l + 1][i] = sigmoid(activation[l + 1][i] + bias[l][i]);
            }
        }
        return activation;
    }

    void trainSample(double[] input, double[] ideal, double[][][] gradW, double[][] gradB) {
        var activation = activation(input);
        var error = new double[activation[weight.length].length];
        for (var i = 0; i < error.length; i++) {
            error[i] = (activation[weight.length][i] - ideal[i]) / (activation[weight.length][i] * (1 - activation[weight.length][i]));
        }
        for (var l = weight.length - 1; l >= 0; l--) {
            var nextError = new double[weight[l][0].length];
            for (var i = 0; i < weight[l].length; i++) {
                error[i] *= activation[l + 1][i] * (1 - activation[l + 1][i]);
                for (var j = 0; j < weight[l][i].length; j++) {
                    gradW[l][i][j] += error[i] * activation[l][j];
                    nextError[j] += error[i] * weight[l][i][j];
                }
                gradB[l][i] += error[i];
            }
            error = nextError;
        }
    }

    void trainBatch(double[][] inputs, double[][] ideals) {
        var gradW = new double[weight.length][][];
        var gradB = new double[bias.length][];
        for (var l = 0; l < weight.length; l++) {
            gradW[l] = new double[weight[l].length][weight[l][0].length];
            gradB[l] = new double[bias[l].length];
        }
        for (var k = 0; k < inputs.length; k++) {
            trainSample(inputs[k], ideals[k], gradW, gradB);
        }
        for (var l = 0; l < weight.length; l++) {
            for (var i = 0; i < weight[l].length; i++) {
                for (var j = 0; j < weight[l][i].length; j++) {
                    weight[l][i][j] -= gradW[l][i][j] / inputs.length;
                }
                bias[l][i] -= bias[l][i] / inputs.length;
            }
        }
    }
}

public class Main {
    public static void main(String[] args) throws Exception {
        var sc = new Scanner(System.in);
        System.out.println("1. Train the network");
        System.out.println("2. Guess a number");
        System.out.print("Your choice: ");
        var choice = sc.nextInt();
        if (choice == 1) {
            System.out.print("Enter the sizes of the layers: ");
            var s = new Scanner(System.in).nextLine().trim().split("\\s+");
            var shape = new int[s.length];
            for (var l = 0; l < shape.length; l++) {
                shape[l] = Integer.parseInt(s[l]);
            }
            System.out.println("Learning...");
            train(shape);
            System.out.println("Done! Saved to the file.");
        } else if (choice == 2) {
            train(new int[]{15, 12, 12, 10});
            guess();
        }
    }

    public static void train(int[] shape) throws Exception {
        var inputs = new double[][]{
                new double[]{0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1},
                new double[]{1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1},
                new double[]{1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
                new double[]{1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
                new double[]{1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1},
                new double[]{1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1},
                new double[]{1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1},
                new double[]{1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0},
                new double[]{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0},
                new double[]{0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1},
                new double[]{1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1},
                new double[]{0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0},
                new double[]{1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1},
                new double[]{1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                new double[]{1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1},
                new double[]{1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1},
                new double[]{1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1},
                new double[]{1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
                new double[]{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1},
                new double[]{1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},
        };
        var ideals = new double[][]{
                new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                new double[]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                new double[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
        };

        var network = new Network(shape);
        for (var epoch = 0; epoch < 1000; epoch++) {
            network.trainBatch(inputs, ideals);
        }

        new ObjectOutputStream(Files.newOutputStream(Path.of("network"))).writeObject(network);
    }

    public static void guess() throws Exception {
        var sc = new Scanner(System.in).useDelimiter("\\s*");
        System.out.println("Input grid:");
        var input = new double[15];
        for (var j = 0; j < input.length; j++) {
            input[j] = sc.next().equals("X") ? 1 : 0;
        }

        var network = (Network) new ObjectInputStream(Files.newInputStream(Path.of("network"))).readObject();
        var activation = network.activation(input);
        var output = activation[activation.length - 1];

        System.out.print("Output:");
        var digit = 0;
        for (var i = 0; i < output.length; i++) {
            System.out.printf(" (%d, %.3f)", i, output[i]);
            if (output[i] > output[digit]) {
                digit = i;
            }
        }
        System.out.println();
        System.out.printf("This number is %d%n", digit);
    }
}