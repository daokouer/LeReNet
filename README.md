# LeReNet

## Something-something training records

### Experiment 1

Models: I3D 32 frames input non-local and mask non-local. Use I3D non-local affine 32 frames inputs as pre-training model.


| <sub>Model</sub> | <sub>Test Acc</sub> | <sub>Commens</sub> | 
| ------------- | ------------- | ------------- |
| <sub>I3D in paper</sub> | <sub>41.6</sub> | <sub></sub> | 
| <sub>I3D non-local in paper</sub> | <sub>44.4</sub> | <sub></sub> | 
| <sub>I3D 32 input</sub> | <sub>38.7</sub> | <sub>2.9 drop than I3D</sub> | 
| <sub>I3D non-loal 32 input</sub> | <sub>44.05</sub> | <sub>0.35 drop than I3D nonlocal</sub> | 
| <sub>I3D mask nlnet 32 input</sub> | <sub>45.3</sub> | <sub>1.1 increase than I3D non-local</sub> | 

Conclusion: I3D experiment in paper(41.6) should be repeated and achieved.

### Experiment 2

Models: I3D 8 input. Baseline == drop rate 0.8, resize 256*320, crop 224, sample rate 4.


| <sub>Model</sub> | <sub>Final Val</sub> | <sub>Final Train</sub> | <sub>Test Acc </sub>|
| ------------- | ------------- | ------------- | ------------- |
| <sub>Base line</sub> | <sub> 28.64</sub> | <sub>77.86</sub> |  <sub>31.62</sub> | 
| <sub>drop rate 0.6</sub> | <sub>29.43</sub> | <sub> 82.82</sub> |  <sub> 32.23</sub> | 
| <sub>Resize 224*224</sub> | <sub>31.55</sub> | <sub>91.17</sub> |  <sub>30.10</sub> | 
| <sub>Resize 224*280</sub> | <sub>31.11</sub> | <sub>85.77</sub> |  <sub>31.74</sub> | 
| <sub>Resize 240*300</sub> | <sub>30.17</sub> | <sub>81.82</sub> |  <sub>30.45</sub> | 
| <sub>Sample rate 3</sub> | <sub> 31.83</sub> | <sub>72.80</sub> |  <sub>35.71</sub> | 
| <sub>Sample rate 3 Resize 224*280</sub> | <sub>34.12</sub> | <sub>33.72</sub> |  <sub> 35.31</sub> | 

Conclusion: 
1. Drop out rate 0.6 is better than 0.8 (not sure).
2. Sample rate 3 is much better than 4.
3. Small feature map caused overfitting on train&val via  test.

### Experiment 3

Models: I3D 8 input. Baseline == drop rate 0.7, resize 232*290, sample rate 3, Res5 stride 1.


|<sub>Model</sub>|<sub>Best Val</sub>|<sub>Final Val</sub>|<sub>Final Train</sub>|<sub>Test Acc </sub>|<sub>Early Model Acc </sub>|
|------------- | ------------- | ------------- | ------------- |------------- |------------- |
|<sub>Base line</sub>|<sub>32.17</sub>|<sub>31.14</sub>|<sub>88.78</sub>|<sub>33.75</sub>|<sub>34.62(115000)</sub>| 
|<sub>drop rate 0.5</sub>|<sub>32.92</sub>|<sub>32.23</sub>|<sub> 91.62</sub>|<sub> 33.67</sub>|<sub>34.12</sub>|
|<sub>Resize 256*320</sub>|<sub>31.32</sub>|<sub>30.116</sub>|<sub>82.48</sub>|<sub>34.37</sub>|<sub>34.59</sub>| 
|<sub>Resize 232*348</sub>|<sub>30.95</sub>|<sub>30.64</sub>|<sub>80.35</sub>|<sub>34.84</sub>|<sub>35.29</sub>|


### Experiment 4
Models: I3D 32 input.
|<sub> model </sub>|<sub>len</sub>|<sub>drop</sub>|<sub>best val</sub>|<sub>final val</sub>|<sub>final train</sub>|<sub>final test</sub>|
|-----------|-------|--------|------------|-------------|---------------|--------------|
|<sub>256*320</sub>|<sub>32 </sub>|<sub>0.75</sub>|<sub> 37.42  </sub>|<sub>  37.34  </sub>|<sub>  78.57    </sub>|<sub>   43.47  </sub>|
|<sub>256*376</sub>|<sub>32 </sub>|<sub>0.75</sub>|<sub> 35.88  </sub>|<sub>  35.52  </sub>|<sub>  72.86    </sub>|<sub>   43.37  </sub>|
|<sub>232*290</sub>|<sub>28 </sub>|<sub>0.75</sub>|<sub> 39.40  </sub>|<sub>  38.91  </sub>|<sub>  85.12    </sub>|<sub>   42.74  </sub>|
|<sub>232*290</sub>|<sub>32 </sub>|<sub>0.7 </sub>|<sub> 39.23  </sub>|<sub>  39.23  </sub>|<sub>  85.70    </sub>|<sub>   42.94  </sub>|
|<sub>232*290</sub>|<sub>32 </sub>|<sub>0.85</sub>|<sub> 39.08  </sub>|<sub>  39.08  </sub>|<sub>  81.88    </sub>|<sub>   42.79  </sub>|
|<sub>232*348</sub>|<sub>32 </sub>|<sub>0.75</sub>|<sub> 38.14  </sub>|<sub>  37.88  </sub>|<sub>  76.88    </sub>|<sub>   44.44  </sub>|
|<sub>224*360</sub>|<sub>32 </sub>|<sub>0.75</sub>|<sub> 38.18  </sub>|<sub>  37.37  </sub>|<sub>  85.86    </sub>|<sub>   44.51  </sub>|


