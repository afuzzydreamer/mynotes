# Python

## Funções Embutidas

## Estrutura de dados

## Strings

   string.split(...)
   .translate
   .mekatrans
   .punctuation

### Listas:

* **list.append(l):**

      x = [1, 2, 3]
      x.append([4, 5])
      print (x)
      >>[1, 2, 3, [4, 5]]

* **list.extend(l):**

      x = [1, 2, 3]
      x.append([4, 5])
      print (x)
      >>[1, 2, 3, 4, 5]

* **list.insert(i, x)**
* **list.remove(x)**

### Dicionários:

* **dicionario.pop(key, 0)**

## Bibliotecas Básicas

### Adicionando Biblioteca em outra pasta.

sys.path.append("../pasta")

### math
* math.ceil(x)
* math.floor(x)
* math.exp(x)
* math.log(c, base)
* math.sqrt(x)
* math.pi
* math.e

### urllib
    import urllib
    response = urllib.urlopen('http://localhost:8080/')
    print 'RESPONSE:', response
    print 'URL     :', response.geturl()

    headers = response.info()
    print 'DATE    :', headers['date']
    print 'HEADERS :'
    print '---------'
    print headers

    data = response.read()
    print 'LENGTH  :', len(data)
    print 'DATA    :'
    print '---------'
    print data

### math.random
*  **random.randrange(stop)**
*  **random.randrange(start, stop[, step]):**
    Return a randomly selected element from range(start, stop, step). This is equivalent to choice(range(start, stop, step)), but doesn’t actually build a range object.

* **random.randint(a, b):** Return a random integer N such that a <= N <= b.


* **random.choice(seq):**   Return a random element from the non-empty sequence seq. If seq is empty, raises IndexError.

* ** random.shuffle(x[, random]):** Shuffle the sequence x in place. The optional argument random is a 0-argument function returning a random float in [0.0, 1.0); by default, this is the function random().

    Note that for even rather small len(x), the total number of permutations of x is larger than the period of most random number generators; this implies that most permutations of a long sequence can never be generated.

* **random.sample(population, k):** Returns a new list containing elements from the population while leaving the original population unchanged. The resulting list is in selection order so that all sub-slices will also be valid random samples. This allows raffle winners (the sample) to be partitioned into grand prize and second place winners (the subslices).

**random.random():** Return the next random floating point number in the range [0.0, 1.0).

**random.uniform(a, b):** The end-point value b may or may not be included in the range depending on floating-point rounding in the equation a + (b-a) * random().

### os


* **os.stats(path):**

* ** os.stat(path):**

   Perform the equivalent of a stat() system call on the given path. (This function follows symlinks; to stat a symlink use lstat().)

   The return value is an object whose attributes correspond to the members of the stat structure, namely:

       st_mode - protection bits,
       st_ino - inode number,
       st_dev - device,
       st_nlink - number of hard links,
       st_uid - user id of owner,
       st_gid - group id of owner,
       st_size - size of file, in bytes,
       st_atime - time of most recent access,
       st_mtime - time of most recent content modification,
       st_ctime - platform dependent; time of most recent metadata change on Unix, or the time of creation on Windows)

### os.path

* **os.path.exists(path)**
Return True if path refers to an existing path. Returns False for broken symbolic links. On some platforms, this function may return False if permission is not granted to execute os.stat() on the requested file, even if the path physically exists.

* **os.path.isfile(path)**
* **os.path.isdir(path)**
* **os.path.join(path, paths):**

 Join one or more path components intelligently. The return value is the concatenation of path and any members of paths with exactly one directory separator (os.sep) following each non-empty part except the last, meaning that the result will only end in a separator if the last part is empty. If a component is an absolute path, all previous components are thrown away and joining continues from the absolute path component.

 On Windows, the drive letter is not reset when an absolute path component (e.g., r'\foo') is encountered. If a component contains a drive letter, all previous components are thrown away and the drive letter is reset. Note that since there is a current directory for each drive, os.path.join("c:", "foo") represents a path relative to the current directory on drive C: (c:foo), not c:\foo.



* **os.remove(path):**

 Remove (delete) the file path. If path is a directory, OSError is raised; see rmdir() below to remove a directory. This is identical to the unlink() function documented below. On Windows, attempting to remove a file that is in use causes an exception to be raised; on Unix, the directory entry is removed but the storage allocated to the file is not made available until the original file is no longer in use.

* **os.removedirs(path):**

    Remove directories recursively. Works like rmdir() except that, if the leaf directory is successfully removed, removedirs() tries to successively remove every parent directory mentioned in path until an error is raised (which is ignored, because it generally means that a parent directory is not empty). For example, os.removedirs('foo/bar/baz') will first remove the directory 'foo/bar/baz', and then remove 'foo/bar' and 'foo' if they are empty. Raises OSError if the leaf directory could not be successfully removed.
Availability: Unix, Windows.

* **os.rename(src, dst):**

   Rename the file or directory src to dst. If dst is a directory, OSError will be raised. On Unix, if dst exists and is a file, it will be replaced silently if the user has permission. The operation may fail on some Unix flavors if src and dst are on different filesystems. If successful, the renaming will be an atomic operation (this is a POSIX requirement). On Windows, if dst already exists, OSError will be raised even if it is a file; there may be no way to implement an atomic rename when dst names an existing file.

### Files

file.seek(n)

### zipfile

* **class zipfile.ZipFile(file[, mode[, compression[, allowZip64]]]):**


* **ZipFile.close():** Close the archive file. You must call close() before exiting your program or essential records will not be written.

* **ZipFile.open(name[, mode[, pwd]]):** Extract a member from the archive as a file-like object (ZipExtFile). name is the name of the file in the archive, or a ZipInfo object. The mode parameter, if included, must be one of the following: 'r' (the default), 'U', or 'rU'. Choosing 'U' or 'rU' will enable universal newline support in the read-only object. pwd is the password used for encrypted files. Calling open() on a closed ZipFile will raise a RuntimeError.

* **ZipFile.namelist():** Return a list of archive members by name.

* **ZipFile.read(name[, pwd]):**  Return the bytes of the file name in the archive. name is the name of the file in the archive, or a ZipInfo object. The archive must be open for read or append. pwd is the password used for encrypted files and, if specified, it will override the default password set with setpassword(). Calling read() on a closed ZipFile will raise a RuntimeError.

* **ZipFile.write(filename[, arcname[, compress_type]]):**  Write the file named filename to the archive, giving it the archive name arcname (by default, this will be the same as filename, but without a drive letter and with leading path separators removed). If given, compress_type overrides the value given for the compression parameter to the constructor for the new entry. The archive must be open with mode 'w' or 'a' – calling write() on a ZipFile created with mode 'r' will raise a RuntimeError. Calling write() on a closed ZipFile will raise a RuntimeError.

    with zipfile.ZipFile(filename) as f:
       data = f.read(f.namelist()[0])).split()
    return data


### Pickle/cPickle

The pickle module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure. “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream is converted back into an object hierarchy. Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” [1] or “flattening”, however, to avoid confusion, the terms used here are “pickling” and “unpickling”.

The pickle module has an optimized cousin called the cPickle module. As its name implies, cPickle is written in C, so it can be up to 1000 times faster than pickle. However it does not support subclassing of the Pickler() and Unpickler() classes, because in cPickle these are functions, not classes. Most applications have no need for this functionality, and can benefit from the improved performance of cPickle. Other than that, the interfaces of the two modules are nearly identical; the common interface is described in this manual and differences are pointed out where necessary. In the following discussions, we use the term “pickle” to collectively describe the pickle and cPickle modules.


* **pickle.dump(obj, file[, protocol]):**
Write a pickled representation of obj to the open file object file. This is equivalent to Pickler(file, protocol).dump(obj).

 If the protocol parameter is omitted, protocol 0 is used. If protocol is specified as a negative value or HIGHEST_PROTOCOL, the highest protocol version will be used.

 file must have a write() method that accepts a single string argument. It can thus be a file object opened for writing, a StringIO object, or any other custom object that meets this interface.

* **pickle.load(file):**
 Read a string from the open file object file and interpret it as a pickle data stream, reconstructing and returning the original object hierarchy. This is equivalent to Unpickler(file).load().

  file must have two methods, a read() method that takes an integer argument, and a readline() method that requires no arguments. Both methods should return a string. Thus file can be a file object opened for reading, a StringIO object, or any other custom object that meets this interface.

  This function automatically determines whether the data stream was written in binary mode or not.

* **pickle.dumps(obj[, protocol]):**
  Return the pickled representation of the object as a string, instead of writing it to a file.

  If the protocol parameter is omitted, protocol 0 is used. If protocol is specified as a negative value or HIGHEST_PROTOCOL, the highest protocol version will be used.

  Changed in version 2.3: The protocol parameter was added.

* **pickle.loads(string):**
  Read a pickled object hierarchy from a string. Characters in the string past the pickled object’s representation are ignored.

### time
   * t0 = time()
   * t1 = time()
   * inter_s = t1 - t0
