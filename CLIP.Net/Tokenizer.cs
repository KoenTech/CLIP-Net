using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace CLIP.Net
{

    public class Tokenizer
    {
        Dictionary<int, char> byteEncoder;
        Dictionary<char, int> byteDecoder;
        Dictionary<string, int> Encoder;
        Dictionary<int, string> Decoder;

        Dictionary<(string, string), int> BpeRanks;
        Dictionary<string, string> Cache;

        Regex Pattern = new Regex(@"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", RegexOptions.Compiled | RegexOptions.IgnoreCase);

        #region static methods
        // 
        //     Returns list of utf-8 byte and a corresponding list of unicode strings.
        //     The reversible bpe codes work on unicode strings.
        //     This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        //     When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        //     This is a signficant percentage of your normal, say, 32K bpe vocab.
        //     To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        //     And avoids mapping to whitespace/control characters the bpe code barfs on.
        //     
        static int ord(char c)
        {
            return (int)c;
        }
        static Dictionary<int, char> bytes_to_unicode()
        {
            var bs = Enumerable.Range(ord('!'), ord('~') + 1 - ord('!')).Concat(Enumerable.Range(ord('¡'), ord('¬') + 1 - ord('¡'))).Concat(Enumerable.Range(ord('®'), ord('ÿ') + 1 - ord('®'))).ToList();
            var cs = bs.ToList();
            var n = 0;
            foreach (var b in Enumerable.Range(0, (int)Math.Pow(2, 8)))
            {
                if (!bs.Contains(b))
                {
                    bs.Add(b);
                    cs.Add((int)(Math.Pow(2, 8) + n));
                    n += 1;
                }
            }
            return Enumerable.Range(0, bs.Count).ToDictionary(x => bs[x], x => cs.Select(o => (char)o).ToList()[x]);
        }

        // Return set of symbol pairs in a word.
        //     Word is represented as tuple of symbols (symbols being variable-length strings).
        //     
        static List<(string, string)> get_pairs(string[] word)
        {
            var pairs = new List<(string, string)>();
            var prev_char = word[0];
            foreach (var character in word[1..])
            {
                pairs.Add((prev_char.ToString(), character.ToString()));
                prev_char = character;
            }
            return pairs;
        }

        static string basic_clean(string text)
        {
            //text = ftfy.fix_text(text);
            //text = html.unescape(html.unescape(text));
            return text.Trim();
        }

        static string whitespace_clean(string text)
        {
            return Regex.Replace(text, @"/\s+/g", " ").Trim();
        }
        #endregion

        public Tokenizer(string vocabPath)
        {
            byteEncoder = bytes_to_unicode();
            byteDecoder = bytes_to_unicode().ToDictionary(x => x.Value, x => x.Key);

            var merges = File.ReadAllLines(vocabPath, Encoding.UTF8);
            merges = merges[1..(((49152  -  256)  -  2)  +  1)];
            var merges2 = merges.Select(x => x.Split(" ")).ToList();
            var vocab = bytes_to_unicode().Values.Select(x => x.ToString()).ToList();
            vocab.AddRange(vocab.Select(x => x+"</w>").ToArray());
            foreach (var merge in merges2)
            {
                vocab.Add(string.Join("", merge));
            }
            vocab.AddRange(new string[] {
                    "<|startoftext|>",
                    "<|endoftext|>"
                });

            Encoder = Enumerable.Range(0, vocab.Count()).ToDictionary(x => vocab[x], x => x);
            Decoder = this.Encoder.ToDictionary(x => x.Value, x => x.Key);
            BpeRanks = Enumerable.Range(0, merges2.Count()).ToDictionary(x => (merges2[x][0], merges2[x][1]), x => x);
            Cache = new Dictionary<string, string> {
                    {
                        "<|startoftext|>",
                        "<|startoftext|>"},
                    {
                        "<|endoftext|>",
                        "<|endoftext|>"}};
        }

        public string BytePairEncode(string token)
        {
            if (Cache.ContainsKey(token))
            {
                return Cache[token];
            }

            var word = token.ToCharArray().Select(x => x.ToString()).ToArray();
            word[word.Length-1] += "</w>";
            var pairs = get_pairs(word);
            if (pairs.Count < 1)
            {
                return token + "</w>";
            }
            while (true)
            {
                var bigRam = pairs.MinBy(x => BpeRanks.GetValueOrDefault(x, int.MaxValue));
                if (!BpeRanks.ContainsKey(bigRam)) break;

                (string first, string second) = bigRam;

                var new_word = new List<string>();
                var i = 0;
                while (i < word.Length)
                {
                    int j = Array.IndexOf(word, first, i);

                    if (j == -1)
                    {
                        new_word.AddRange(word[i..]);
                        break;
                    }

                    new_word.AddRange(word[i..j]);
                    i = j;

                    if (word[i].ToString() == first && i < (word.Length-1) && word[i+1].ToString() == second)
                    {
                        new_word.Add(first+second);
                        i+=2;
                    }
                    else
                    {
                        new_word.Add(word[i].ToString());
                        i++;
                    }
                }
                word = new_word.ToArray();
                if (word.Length == 1)
                {
                    break;
                }
                else
                {
                    pairs = get_pairs(word);
                }
            }
            var result = string.Join(" ", word);
            Cache[token] = result;
            return result;
        }

        public int[] Encode(string text)
        {
            var bpe_tokens = new List<int>();
            text = whitespace_clean(text).ToLower();
            foreach (var token in Pattern.Matches(text).Select(x => x.Value))
            {
                bpe_tokens.AddRange(BytePairEncode(token).Split(' ').Select(x => Encoder[x]));
            }
            return bpe_tokens.ToArray();
        }

        public string Decode(int[] tokens)
        {
            var text = string.Join("", tokens.Select(x => Decoder[x])).Replace("</w>", " ");
            return text;
        }

        public int[] EncodeForCLIP(string text)
        {
            var tokens = this.Encode(text);
            var data = tokens.Prepend(49406).Append(49407);
            while (data.Count() < 77)
            {
                data = data.Append(0);
            }
            return data.ToArray();
        }
    }
}
