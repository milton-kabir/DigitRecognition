//package recognition;
package com.kabir.milton;

import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        // write your code here
        System.out.println("Input grid:");
        String[] ar = new String[3];
        Scanner sc = new Scanner(System.in);
        for (int i = 0; i < 3; i++) {
            ar[i] = sc.nextLine();
        }
        int s = 0;
        int[] br = {2, 1, 2, 4, -4, 4, 2, -1, 2, -5};
        for (int i = 0, k = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++, k++) {
                if (ar[i].charAt(j) == 'X') {
                    s += br[k];
                }
            }
        }
        s += br[9];
        if (s > 0) {
            System.out.println("This number is 0");
        } else {
            System.out.println("This number is 1");
        }
    }
}
