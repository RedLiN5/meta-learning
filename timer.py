def timer(start,end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time Cost: {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))
