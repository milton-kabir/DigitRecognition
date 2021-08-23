//package recognition;
package com.kabir.milton;

import java.util.Scanner;

public class Main{
    static Scanner sc = new Scanner(System.in);

    public static int[] getNeuron() {
        int[] neuron = new int[15];

        for (int i = 0, k = 0; i < 5; i++) {
            String temp = sc.next();
            System.out.println(temp);
            for (int j = 0; j < 3; j++) {
                neuron[k++] = temp.charAt(j) == 'X' ? 1 : 0;
            }
        }
        return neuron;
    }

    public static int getWeightedSum(int[] neuron) {
        int max = 0;
        int[] bias = {-1, 6, 1, 0, 2, 0, -1, 3, -2, -1};
        int temp = 0;
        int number = 0;
        int[][] weights = {
                {1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1},			//0
                {-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1},	//1
                {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1},			//2
                {1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1},			//3
                {1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1},		//4
                {1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1},			//5
                {1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1},			//6
                {1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1},		//7
                {1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1},			//8
                {1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1}			//9
        };

        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 15; k++) {
                temp += weights[j][k] * neuron[k];
            }
            temp += bias[j];
            if (temp > max) {
                max = temp;
                number = j;
            }
            temp = 0;
        }
        return number;
    }

    public static void main(String[] args) {
        int[] neuron = getNeuron();
        int result = getWeightedSum(neuron);
        System.out.println(result);

    }
}