**Digital Image Processing (DIP)** is the process of converting a real-world visual scene into numerical pixel data, processing those numbers, and finally extracting useful information such as intensity, edges, boundaries, and shapes.


# 1

The journey starts with **image acquisition**. :
- A sensor such as CCD or CMOS converts incoming energy into an electrical signal.
- Since the real signal is continuous, **sampling** converts continuous spatial coordinates into discrete pixels,
- while **quantization** converts continuous intensity into finite numerical levels.
- an image becomes a matrix of pixels. For a `k`-bit image, the number of gray levels is `L = 2^k`.
# 2

Once we have pixels, we study their **relationships with neighbouring pixels** 4-neighbourhood, diagonal neighbourhood and 8-neighbourhood. This becomes the foundation for spatial processing If we need to :
- resize an image 
- interpolation esitmates missing pixel value
- nearest neighbour copies the closest value,
- bilinear uses four surrounding pixels with weighted averaging
- and bicubic uses a larger 4×4 neighbourhood.

# 3 

We can then perform direct **pixel operations** 
- addition
- substraction
- averaging
- masking
- inversion
These can combine
- information
- Change brightness. 
- Detect differences, 
- reduce noise, or 
- isolate regions. 
The key idea is that the image is ultimately numerical data so a mathematical operation can manipulate its appearance and information. 

# 4 

- Image enhancement 
- Brightness transformation shifts intensities using `g = f + β`
- contrast transformation stretches or compresses them using `g = αf`
- converts an image into object/background regions
- while gray-level slicing focuses on a selected intensity range
- Histogram stretching uses the available intensity range more effectively.
- Log, exponential and gamma transformations provide different ways of modifying dark and bright regions.

# 5 

- The **histogram** tells us how pixel intensities are distributed. but not where those pixels are located.
- Histogram equalization uses the CDF to redistribute intensities and improve contrast
- if different parts of an image need different enhancement,
- **AHE**; because AHE can amplify noise,
- **CLAHE** limits the contrast,
- redistributes excess histogram values, and combines the enhanced regions smoothly.

# 6 

- we move from individual pixels to **neighbourhood-based spatial filtering**.
- A kernel looks at surrounding pixels and calculates a new output value.
- **Low-pass filters** suppress rapid variations and are therefore used for smoothing/noise reduction.
- **High-pass filters** emphasize rapid intensity changes and are used for sharpening and edge detection.
- Importantly, the professor's notes define frequency through **variation/variance**, not simply brightness

# 7

- For smoothing, the **mean filter** averages neighbouring pixels but can blur edges.
- The **Gaussian filter** gives greater weight to nearby pixels, with σ controlling the amount of smoothing.
- The **median filter** removes outliers and is particularly useful for salt-and-pepper noise while preserving edges better.
- This gives an important processing principle:
- Smooth → reduce noise → then detect edges.

# 8

- After smoothing, we can search for **rapid intensity changes**, which are edges
- First derivatives measure intensity change, while the **gradient** gives both edge strength and direction.
- **Prewitt** approximates horizontal/vertical derivatives, while **Sobel** does the same with additional centre weighting, giving the professor's shortcut:
- **Sobel = derivative + smoothing.**

# 9 

- The **Laplacian** uses second derivatives to detect rapid changes in the gradient.
- Since it is sensitive to noise, we combine Gaussian smoothing with the Laplacian to form **LoG (Laplacian of Gaussian)**. Zero crossing occurs where the Laplacian changes sign.

# 10 

- Detected edges can then be combined into **boundaries** which describe enclosed objects 
- From boundaries we can obtain region properties such as area and perimeter, detect corners, and characterize shapes. Circularity, for example, is `4πA/P²`. This creates the progression:
- **Pixels → Edges → Boundaries → Regions → Shapes.**

- Finally, the **Hough Transform** detects geometric structures such as lines and circles.
- Edge pixels vote for possible shapes in a parameter space; the strongest accumulation indicates the detected shape.
- For lines: `ρ = x cosθ + y sinθ`
- For circles: `(x−a)² + (y−b)² = r²`
- The notes then connect DIP to **CNNs**: traditional DIP uses manually designed kernels to detect/smooth features, while CNNs learn filters automatically. The basic idea remains the same:

**Image → neighbourhood → convolution/filter → features → higher-level recognition.**

**Real scene**  
→ Sensor  
→ Sampling + Quantization  
→ Digital Image  
→ Pixels + Neighbours  
→ Interpolation / Pixel Operations  
→ Intensity Enhancement  
→ Histogram Processing  
→ Spatial Filtering  
→ Smoothing  
→ Edge Detection  
→ Boundaries / Regions  
→ Shape Detection  
→ Hough Transform  
→ CNN Feature Learning

Start with raw pixel values, understand their relationships, modify them when necessary, then progressively extract higher-level visual information from them.

Real Scene → Sensor → Sampling + Quantization → Digital Image → Pixels/Neighbours → Resampling & Pixel Operations → Intensity Enhancement → Histogram Processing → Spatial Filtering → Smoothing → Edge Detection → Boundaries → Regions/Shapes → Hough Transform → CNN Feature Learning

- **Image Fundamentals:** Convert the real-world scene into a digital image using sensors, sampling and quantization; understand pixels, resolution, gray levels and neighbourhoods.
- **Resampling & Pixel Operations:** Resize images using nearest-neighbour, bilinear or bicubic interpolation, and manipulate images through addition, subtraction, averaging, masking and inversion.
- **Intensity Enhancement:** Improve brightness and contrast using intensity transformations such as brightness, contrast, thresholding, gray-level slicing, stretching, log, exponential and gamma transformations.
- **Histogram Processing:** Study how intensities are distributed and improve contrast using histogram stretching, equalization, AHE and CLAHE.
- **Spatial Filtering:** Use kernels and neighbouring pixels to process images; low-pass filters smooth/reduce noise, while high-pass filters emphasize details and edges.
- **Smoothing:** Remove unwanted variations before further processing using mean, Gaussian and median filters, with median being particularly useful for outlier/salt-and-pepper noise.
- **Sharpening & Edge Detection:** Detect intensity changes using gradients, Prewitt, Sobel and Laplacian; LoG combines Gaussian smoothing with Laplacian to reduce noise sensitivity.
- **Edges, Boundaries & Shapes:** Convert detected edges into boundaries and regions, then describe objects using area, perimeter, corners, circularity and other shape properties.
- **Hough Transform:** Detect geometric structures such as lines and circles by letting edge pixels vote in a parameter/accumulator space.
- **DIP → CNN:** CNNs extend the same pixel-neighbourhood-convolution idea, but instead of manually designing filters, they learn features such as edges, textures and shapes automatically.