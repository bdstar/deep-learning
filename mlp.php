<?php

// Define the neural network class
class MLP
{
    private $inputLayerSize;
    private $hiddenLayerSize;
    private $outputLayerSize;
    private $weightsInputHidden;
    private $weightsHiddenOutput;

    // Constructor to initialize the neural network
    public function __construct($inputLayerSize, $hiddenLayerSize, $outputLayerSize)
    {
        $this->inputLayerSize = $inputLayerSize;
        $this->hiddenLayerSize = $hiddenLayerSize;
        $this->outputLayerSize = $outputLayerSize;

        // Initialize weights randomly (you might want to initialize them differently)
        $this->weightsInputHidden = $this->initializeWeights($this->inputLayerSize, $this->hiddenLayerSize);
        $this->weightsHiddenOutput = $this->initializeWeights($this->hiddenLayerSize, $this->outputLayerSize);
    }

    // Function to initialize weights randomly
    private function initializeWeights($rows, $cols)
    {
        $weights = array();

        for ($i = 0; $i < $rows; $i++) {
            for ($j = 0; $j < $cols; $j++) {
                $weights[$i][$j] = rand() / getrandmax(); // Random initialization (better methods exist)
            }
        }

        return $weights;
    }

    // Sigmoid activation function
    private function sigmoid($x)
    {
        return 1 / (1 + exp(-$x));
    }

    // Function to perform forward pass
    public function forward($inputs)
    {
        // Calculate output of the hidden layer
        $hiddenOutput = array();
        for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
            $sum = 0;
            for ($j = 0; $j < $this->inputLayerSize; $j++) {
                $sum += $inputs[$j] * $this->weightsInputHidden[$j][$i];
            }
            $hiddenOutput[$i] = $this->sigmoid($sum);
        }

        // Calculate final output
        $finalOutput = array();
        for ($i = 0; $i < $this->outputLayerSize; $i++) {
            $sum = 0;
            for ($j = 0; $j < $this->hiddenLayerSize; $j++) {
                $sum += $hiddenOutput[$j] * $this->weightsHiddenOutput[$j][$i];
            }
            $finalOutput[$i] = $this->sigmoid($sum);
        }

        return $finalOutput;
    }

    // Function to train the neural network
    public function train($inputs, $targets, $epochs, $learningRate)
    {
        for ($epoch = 0; $epoch < $epochs; $epoch++) {
            // Forward pass
            $hiddenOutput = array();
            for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
                $sum = 0;
                for ($j = 0; $j < $this->inputLayerSize; $j++) {
                    $sum += $inputs[$j] * $this->weightsInputHidden[$j][$i];
                }
                $hiddenOutput[$i] = $this->sigmoid($sum);
            }

            $finalOutput = array();
            for ($i = 0; $i < $this->outputLayerSize; $i++) {
                $sum = 0;
                for ($j = 0; $j < $this->hiddenLayerSize; $j++) {
                    $sum += $hiddenOutput[$j] * $this->weightsHiddenOutput[$j][$i];
                }
                $finalOutput[$i] = $this->sigmoid($sum);
            }

            // Backpropagation
            $outputError = array();
            for ($i = 0; $i < $this->outputLayerSize; $i++) {
                $outputError[$i] = $targets[$i] - $finalOutput[$i];
            }

            $hiddenError = array();
            for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
                $sum = 0;
                for ($j = 0; $j < $this->outputLayerSize; $j++) {
                    $sum += $outputError[$j] * $this->weightsHiddenOutput[$i][$j];
                }
                $hiddenError[$i] = $hiddenOutput[$i] * (1 - $hiddenOutput[$i]) * $sum;
            }

            // Update weights
            for ($i = 0; $i < $this->inputLayerSize; $i++) {
                for ($j = 0; $j < $this->hiddenLayerSize; $j++) {
                    $this->weightsInputHidden[$i][$j] += $learningRate * $inputs[$i] * $hiddenError[$j];
                }
            }

            for ($i = 0; $i < $this->hiddenLayerSize; $i++) {
                for ($j = 0; $j < $this->outputLayerSize; $j++) {
                    $this->weightsHiddenOutput[$i][$j] += $learningRate * $hiddenOutput[$i] * $outputError[$j];
                }
            }
        }
    }
}

// Example usage
$inputLayerSize = 2;
$hiddenLayerSize = 3;
$outputLayerSize = 1;

$mlp = new MLP($inputLayerSize, $hiddenLayerSize, $outputLayerSize);

// Training data
$trainingInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
$trainingTargets = [[0], [1], [1], [0]];

// Train the neural network
$mlp->train($trainingInputs[0], $trainingTargets[0], 10000, 0.1);

// Test the neural network
foreach ($trainingInputs as $index => $input) {
    $output = $mlp->forward($input);
    echo "Input: " . implode(", ", $input) . " | Output: " . round($output[0]) . "<br>";
}
