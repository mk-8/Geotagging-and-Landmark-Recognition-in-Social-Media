# Geotagging-and-Landmark-Recognition-in-Social-Media

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, I built models to automatically predict the location of the image based on any landmarks depicted in the image. I went through the machine learning design process end-to-end: performed data preprocessing, designed and trained CNNs, compared the accuracy of different CNNs, and used my own images to heuristically evaluate the best CNN.
