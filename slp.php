<?php
$input_layer = 2;
$threshold = 1;
$learning_rate = 0.5;

#train dataset AND gate
/*$xs = array(
    array(0,0,0),
    array(0,1,0),
    array(1,0,0),
    array(1,1,1)
);*/

#train dataset OR gate
/*$xs = array(
    array(0,0,0),
    array(0,1,1),
    array(1,0,1),
    array(1,1,1)
);*/

#train dataset coordinate point
$xs = array(
    array(1,1,1),
    array(0,4,1),
    array(5,7,1),
    array(6,4,1),
    array(-1,-1,0),
    array(-7,-4,0),
    array(-5,-7,0),
    array(-3,-4,0),
    array(9,5,1),
    array(7,4,1),
    array(-4,-5,0),
    array(-3,-7,0)    
);

#Initalize weight data for input layer
$w = array(1.2, 0.6);

function Single_Layer_Perceptron_training($xs, $w){
    global $input_layer;
    global $threshold;
    global $learning_rate;

    foreach ($xs as $k => $x) {
        # echo $x[0]."&".$x[1]."=".$x[2];
    
        $wx=0;
        for ($i=0; $i < $input_layer; $i++) { 
            $wx += $x[$i] * $w[$i];
        }
        # echo ":wx=".$wx;
    
        $output = ($wx<$threshold) ? 0 : 1;
        # echo ":output=".$output;  
    
        if($output != $x[2]){
            # echo "<br>*MISTAKE*<br>";
            for ($i=0; $i < $input_layer; $i++) { 
                $w[$i] = $w[$i] + $learning_rate * ($x[2] - $output)*$x[$i];
                # echo "w[".$i."]:".$w[$i]."<br>";
            }
            Single_Layer_Perceptron_training($xs, $w);
        }
        # echo "<br>============<br>";
    }
    return $w;
}

function Single_Layer_Perceptron_prediction($x, $w){
    global $input_layer;
    global $threshold;
    global $learning_rate;

    $wx=0;
    for ($i=0; $i < $input_layer; $i++)
        $wx += $x[$i] * $w[$i];

    $output = ($wx<$threshold) ? 0 : 1;
    return $output;
}

$weight = Single_Layer_Perceptron_training($xs, $w);


$input_data = array(-5,-7);
echo Single_Layer_Perceptron_prediction($input_data, $weight);
?>