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
