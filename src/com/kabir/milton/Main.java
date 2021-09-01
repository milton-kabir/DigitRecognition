//package recognition;
package com.kabir.milton;

import java.io.*;
import java.util.*;

class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 7L;

    private static final int numInputNeurons = 15;

    private static final int numOutputNeurons = 10;

    private final double[][] weights;

    NeuralNetwork() {
        weights = new double[numOutputNeurons][numInputNeurons + 1];
    }

    public void learn() {
        NeuralNetworkUtils.gaussianRandom(weights);

        for (int generation = 0; generation < 1000; generation++) {
            nextGeneration();
        }
    }

    private void nextGeneration() {
        for (int outputNeuron = 0; outputNeuron < numOutputNeurons; outputNeuron++) {

            double[] deltaWmean = new double[numInputNeurons];

            for (int number = 0; number < numOutputNeurons; number++) {
                double sum = 0;
                for (int i = 0; i < numInputNeurons; i++) {
                    sum += NeuralNetworkUtils.idealInputs[number][i] *
                            weights[outputNeuron][i];
                }
                sum += weights[outputNeuron][numInputNeurons];

                double o = NeuralNetworkUtils.sigmoid(sum);

                int idealValue = number == outputNeuron ? 1 : 0;
                for (int i = 0; i < numInputNeurons; i++) {
                    double delta = NeuralNetworkUtils.ETA *
                            NeuralNetworkUtils.idealInputs[number][i] *
                            (idealValue - o);
                    deltaWmean[i] += delta;
                }
            }

            for (int i = 0; i < numInputNeurons; i++) {
                weights[outputNeuron][i] += deltaWmean[i] / numOutputNeurons;
            }

        }
    }

    public int recognize(List<List<Integer>> matrix) {
        List<Integer> inputNeurons = NeuralNetworkUtils.flatten(matrix);

        List<Double> outputNeurons = new LinkedList<>();

        for (int i = 0; i < numOutputNeurons; i++) {
            double sum = 0;
            for (int j = 0; j < numInputNeurons; j++) {
                sum += inputNeurons.get(j) * weights[i][j];
            }
            sum += weights[i][numInputNeurons];

            outputNeurons.add(sum);
        }

        return outputNeurons.indexOf(outputNeurons.
                stream().
                mapToDouble(v -> v).
                max().getAsDouble());
    }

}

abstract class NeuralNetworkUtils {

    public static final double[][] idealInputs = {
            {1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1},  //0
            {0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0},  //1
            {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1},  //2
            {1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1},  //3
            {1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1},  //4
            {1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1},  //5
            {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1},  //6
            {1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},  //7
            {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1},  //8
            {1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1}   //9
    };

    public static final double ETA = 0.5;

    public static void gaussianRandom(double[][] weights) {
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static List<Integer> flatten(List<List<Integer>> nestedList) {
        List<Integer> ls = new LinkedList<>();
        nestedList.forEach(ls::addAll);
        return ls;
    }

}

class SerializationUtils {
    /**
     * Serialize the given object to the file
     */
    public static void serialize(Object obj, String fileName) throws IOException {
        FileOutputStream fos = new FileOutputStream(fileName);
        BufferedOutputStream bos = new BufferedOutputStream(fos);
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeObject(obj);
        oos.close();
    }

    /**
     * Deserialize to an object from the file
     */
    public static Object deserialize(String fileName) throws IOException, ClassNotFoundException {
        FileInputStream fis = new FileInputStream(fileName);
        BufferedInputStream bis = new BufferedInputStream(fis);
        ObjectInputStream ois = new ObjectInputStream(bis);
        Object obj = ois.readObject();
        ois.close();
        return obj;
    }
}

public class Main {

    public static Scanner scanner = new Scanner(System.in);

    public static void main(String[] args) {

        System.out.println("1. Learn the network\n" +
                "2. Guess a number");
        System.out.print("Your choice: ");

        int choice = Integer.parseInt(scanner.nextLine());

        switch (choice) {
            case 1: {
                learning();
                break;
            }
            case 2: {
                learning();
                guessing();
                break;
            }
        }
    }

    public static void learning() {

        NeuralNetwork network = new NeuralNetwork();

        System.out.println("Learning...");

        network.learn();
        try {
            SerializationUtils.serialize(network, "network.data");
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Done! Saved to the file.");
    }

    public static void guessing() {

        NeuralNetwork network = new NeuralNetwork();
        try {
            network = (NeuralNetwork) SerializationUtils.deserialize("network.data");
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        System.out.println("Input grid:");
        List<List<Integer>> matrix = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            String row = scanner.nextLine();
            List<Integer> rowAsListOfIntegers = new ArrayList<>();
            for (char c : row.toCharArray()) {
                rowAsListOfIntegers.add(c == 'X' ? 1 : 0);
            }
            matrix.add(rowAsListOfIntegers);
        }

        System.out.println("This number is " + network.recognize(matrix));
    }


}