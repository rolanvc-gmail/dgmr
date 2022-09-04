def space_to_depth(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and width of tensor must be divisible by scale_factor.')
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape([num, ch, new_height, scale_factor, new_width, scale_factor]) # divide by 2 the height and width then create four stacks
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, scale_factor*scale_factor, ch, new_height, new_width])
    return tensor


def main():
    pass


if __name__ == "__main__":
    main()
