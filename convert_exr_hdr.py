import OpenImageIO as oiio

inp = oiio.ImageInput.open('./assets/HDRi/pav_studio_03_4k.exr')
spec = inp.spec()
pixels = inp.read_image()
inp.close()

out = oiio.ImageOutput.create('./assets/HDRi/pav_studio_03_4k.hdr')
out.open('./assets/HDRi/pav_studio_03_4k.hdr', spec)
out.write_image(pixels)
out.close()
