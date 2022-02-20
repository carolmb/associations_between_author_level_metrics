import bz2

if __name__ == '__main__':
    f = bz2.open("mag-data/02.Authors.nt.bz2", "rb")
    data = f.read()
    print(data[:1])
    obj = bz2.BZ2DeCompressor()
    decompressed = obj.decompress(data[:1])
    print(decompressed)