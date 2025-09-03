// ------------------------------
// ГЛАВА 3: Компютърна реализация – C#/.NET 8 Console App (единичен файл)
// Program.cs – логика, формули, извод на резултати.
// ------------------------------
// Какво прави файлът:
// 1) Чете входен CSV: mood,strategy,genre,S_subj,S_bio,n_counts (BG/EN етикети)
// 2) Изчислява p(g|x) с Лаплас α, H~(x), H(G|X), C=1-H/lnK
// 3) S_meta(g|x) = C(x) * (p(g|x) - 1/K) / (1 - 1/K)
// 4) u_{x,g} = w_subj*S_subj + w_bio*S_bio + w_meta*S_meta
// 5) Платежна матрица A (η наказание), репликатор с мутация μ
// 6) Запис: utilities_u.txt, stationary_xstar.txt, top3_by_u_and_xstar.txt,
//           entropy_summary.txt, details_by_state.txt  (NOTE файлове)
// 7) Ако CSV липсва -> съобщение и край; конзолен избор на настроение/стратегия.
// ------------------------------

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace MusicEvo
{
    internal static class Program
    {
        private const double EPS = 1e-12;

        public enum Mood { Happiness, Sadness, Stress, Motivation, Relax }
        public enum Strategy { Amplify, Suppress, Stabilize, Focus, Neutralize, Gradual }
        public enum Genre { Pop, Rock, Classical, HipHop, Jazz, EDM, Ambient, Metal }
        private static int K => Enum.GetValues(typeof(Genre)).Length;

        public enum Conditioning { Mood, MoodStrategy }

        public record Config
        {
            public double Alpha { get; init; } = 1.0;
            public double WSubj { get; init; } = 0.5;
            public double WBio { get; init; } = 0.3;
            public double WMeta { get; init; } = 0.2;
            public double Eta { get; init; } = 0.15;
            public double Tau { get; init; } = 0.5;
            public double Mu { get; init; } = 1e-3;
            public int Steps { get; init; } = 10000;
            public string OutDir { get; init; } = "out";
            public Conditioning Granularity { get; init; } = Conditioning.Mood;
        }

        public sealed class Row
        {
            public Mood Mood { get; init; }
            public Strategy Strategy { get; init; }
            public Genre Genre { get; init; }
            public double SSubj { get; init; }
            public double SBio { get; init; }
            public int NCounts { get; init; }
        }

        // ---------- Етикети BG/EN ----------
        private static readonly Dictionary<string, Mood> MoodMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["happiness"] = Mood.Happiness,
            ["щастие"] = Mood.Happiness,
            ["sadness"] = Mood.Sadness,
            ["тъга"] = Mood.Sadness,
            ["stress"] = Mood.Stress,
            ["стрес"] = Mood.Stress,
            ["motivation"] = Mood.Motivation,
            ["мотивация"] = Mood.Motivation,
            ["relax"] = Mood.Relax,
            ["релакс"] = Mood.Relax,
        };

        private static readonly Dictionary<string, Strategy> StrategyMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["amplify"] = Strategy.Amplify,
            ["усилване"] = Strategy.Amplify,
            ["suppress"] = Strategy.Suppress,
            ["потискане"] = Strategy.Suppress,
            ["stabilize"] = Strategy.Stabilize,
            ["стабилизиране"] = Strategy.Stabilize,
            ["focus"] = Strategy.Focus,
            ["фокусиране"] = Strategy.Focus,
            ["neutralize"] = Strategy.Neutralize,
            ["неутрализиране"] = Strategy.Neutralize,
            ["gradual"] = Strategy.Gradual,
            ["плавна промяна"] = Strategy.Gradual,
        };

        private static readonly Dictionary<string, Genre> GenreMap = new(StringComparer.OrdinalIgnoreCase)
        {
            ["pop"] = Genre.Pop,
            ["поп"] = Genre.Pop,
            ["rock"] = Genre.Rock,
            ["рок"] = Genre.Rock,
            ["classical"] = Genre.Classical,
            ["класика"] = Genre.Classical,
            ["класическа"] = Genre.Classical,
            ["hip-hop"] = Genre.HipHop,
            ["hiphop"] = Genre.HipHop,
            ["hip hop"] = Genre.HipHop,
            ["хип-хоп"] = Genre.HipHop,
            ["jazz"] = Genre.Jazz,
            ["джаз"] = Genre.Jazz,
            ["edm"] = Genre.EDM,
            ["едм"] = Genre.EDM,
            ["електронна"] = Genre.EDM,
            ["ambient"] = Genre.Ambient,
            ["ембиънт"] = Genre.Ambient,
            ["амбийнт"] = Genre.Ambient,
            ["амбИент"] = Genre.Ambient,
            ["metal"] = Genre.Metal,
            ["метъл"] = Genre.Metal
        };

        private static string GenreNameEn(Genre g) => g switch { Genre.HipHop => "Hip-hop", _ => g.ToString() };
        private static string GenreNameBg(Genre g) => g switch
        {
            Genre.Pop => "Поп",
            Genre.Rock => "Рок",
            Genre.Classical => "Класика",
            Genre.HipHop => "Хип-хоп",
            Genre.Jazz => "Джаз",
            Genre.EDM => "EDM",
            Genre.Ambient => "Ембиънт",
            Genre.Metal => "Метъл",
            _ => g.ToString()
        };
        private static string MoodBg(Mood m) => m switch
        {
            Mood.Happiness => "Щастие",
            Mood.Sadness => "Тъга",
            Mood.Stress => "Стрес",
            Mood.Motivation => "Мотивация",
            Mood.Relax => "Релакс",
            _ => m.ToString()
        };
        private static string StratBg(Strategy s) => s switch
        {
            Strategy.Amplify => "Усилване",
            Strategy.Suppress => "Потискане",
            Strategy.Stabilize => "Стабилизиране",
            Strategy.Focus => "Фокусиране",
            Strategy.Neutralize => "Неутрализиране",
            Strategy.Gradual => "Плавна промяна",
            _ => s.ToString()
        };

        // ---------- Математика ----------
        private static double[] Softmax(double[] u, double tau)
        {
            double t = Math.Max(tau, EPS);
            double max = u.Max();
            double[] z = u.Select(v => (v - max) / t).ToArray();
            double[] e = z.Select(Math.Exp).ToArray();
            double sum = e.Sum();
            return e.Select(v => v / Math.Max(sum, EPS)).ToArray();
        }

        private static double ConditionalEntropy(double[] p)
        {
            double h = 0.0;
            foreach (var v in p)
            {
                double q = Math.Max(v, EPS);
                h += -q * Math.Log(q);
            }
            return h;
        }

        // S_meta(g|x) = C(x) * (p(g|x) - 1/K) / (1 - 1/K)
        private static double[] SMetaPerGenre(double[] p)
        {
            double h = ConditionalEntropy(p);
            double c = 1.0 - h / Math.Log(K);
            double center = 1.0 / K;
            double denom = (1.0 - center) + EPS;
            double[] s = new double[K];
            for (int i = 0; i < K; i++) 
                s[i] = c * ((p[i] - center) / denom);
            return s;
        }

        private static double[] PayoffDynamicsStep(double[,] A, double[] x, double mu)
        {
            int n = x.Length;
            var Ax = new double[n];
            for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i, j] * x[j]; Ax[i] = s; }

            double minAx = Ax.Min();
            if (minAx <= 0) for (int i = 0; i < n; i++) Ax[i] = Ax[i] - minAx + EPS;

            double avg = 0; for (int i = 0; i < n; i++) avg += x[i] * Ax[i];
            avg = Math.Max(avg, EPS);

            var next = new double[n];
            for (int i = 0; i < n; i++) next[i] = x[i] * (Ax[i] / avg);

            if (mu > 0)
            {
                double u = 1.0 / n;
                for (int i = 0; i < n; i++) next[i] = (1 - mu) * next[i] + mu * u;
            }

            double sum = next.Sum();
            if (sum <= 0) return Enumerable.Repeat(1.0 / n, n).ToArray();
            for (int i = 0; i < n; i++) next[i] = Math.Max(0, next[i] / sum);
            double norm = next.Sum();
            for (int i = 0; i < n; i++) next[i] /= Math.Max(norm, EPS);
            return next;
        }

        private static double L1(double[] a, double[] b)
        {
            double s = 0; for (int i = 0; i < a.Length; i++) s += Math.Abs(a[i] - b[i]);
            return s;
        }

        private static double[] Normalize(double[] x)
        {
            double s = x.Sum();
            if (s <= 0) return Enumerable.Repeat(1.0 / x.Length, x.Length).ToArray(); // FIX
            return x.Select(v => Math.Max(0, v / s)).ToArray();
        }

        // ---------- Основни стъпки ----------
        private static (double[] u, double[] sMeta, double[] sSubj, double[] sBio, double[] p_gx, double Cx, double Hx)
            UtilitiesFor(IReadOnlyList<Row> rows, Config cfg, Mood mood, Strategy strategy)
        {
            int[] counts = new int[K];
            if (cfg.Granularity == Conditioning.Mood)
                foreach (var r in rows.Where(r => r.Mood == mood)) counts[(int)r.Genre] += r.NCounts;
            else
                foreach (var r in rows.Where(r => r.Mood == mood && r.Strategy == strategy)) counts[(int)r.Genre] += r.NCounts;

            int n = counts.Sum();
            double[] p_gx = new double[K];
            for (int i = 0; i < K; i++)
                p_gx[i] = (counts[i] + cfg.Alpha) / (Math.Max(n, 0) + cfg.Alpha * K);

            double Hx = ConditionalEntropy(p_gx);
            double Cx = 1.0 - Hx / Math.Log(K);
            var sMeta = SMetaPerGenre(p_gx);

            double[] sSubj = new double[K];
            double[] sBio = new double[K];
            foreach (var r in rows.Where(r => r.Mood == mood && r.Strategy == strategy))
            {
                int g = (int)r.Genre;
                sSubj[g] = r.SSubj;
                sBio[g] = r.SBio;
            }

            double[] u = new double[K];
            for (int g = 0; g < K; g++)
                u[g] = cfg.WSubj * sSubj[g] + cfg.WBio * sBio[g] + cfg.WMeta * sMeta[g];

            return (u, sMeta, sSubj, sBio, p_gx, Cx, Hx);
        }

        // Диагонал: u_j; извън диаг.: 0.5(u_j+u_k) - η * |u_j - u_k|
        private static double[,] BuildPayoffMatrix(double[] u, double eta)
        {
            var A = new double[K, K];
            for (int j = 0; j < K; j++)
                for (int k = 0; k < K; k++)
                    A[j, k] = (j == k) ? u[j] : 0.5 * (u[j] + u[k]) - eta * Math.Abs(u[j] - u[k]);
            return A;
        }

        private static double[] Replicator(double[,] A, Config cfg, double[]? x0 = null) // nullable FIX
        {
            double[] x = x0 != null ? Normalize(x0) : Enumerable.Repeat(1.0 / K, K).ToArray();
            var history = new Queue<double[]>();
            double lastAvg = 0; int stableCount = 0;

            for (int t = 0; t < cfg.Steps; t++)
            {
                int n = x.Length;
                var Ax = new double[n];
                for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += A[i, j] * x[j]; Ax[i] = s; }
                double minAx = Ax.Min(); if (minAx <= 0) for (int i = 0; i < n; i++) Ax[i] = Ax[i] - minAx + EPS;
                double avgPayoff = 0; for (int i = 0; i < n; i++) avgPayoff += x[i] * Ax[i];

                var next = PayoffDynamicsStep(A, x, cfg.Mu);

                history.Enqueue((double[])x.Clone());
                if (history.Count > 50) history.Dequeue();

                if (history.Count == 50 && L1(next, history.Peek()) < 1e-7 && stableCount > 100)
                { x = next; break; }

                x = next;
            }
            return Normalize(x);
        }

        // ---------- Вход/изход ----------
        private static List<Row> LoadCsv(string path)
        {
            var rows = new List<Row>();
            using var sr = new StreamReader(path);
            string? header = sr.ReadLine(); // пропускаме заглавието
            while (!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                if (string.IsNullOrWhiteSpace(line)) continue;

                var parts = line.Split(',');
                if (parts.Length < 6) continue;

                string moodStr = parts[0].Trim();
                string stratStr = parts[1].Trim();
                string genreStr = parts[2].Trim();
                string sSubjStr = parts[3].Trim();
                string sBioStr = parts[4].Trim();
                string nStr = parts[5].Trim();

                if (!MoodMap.TryGetValue(moodStr, out var mood))
                    throw new Exception($"Неизвестно настроение: {moodStr}");
                if (!StrategyMap.TryGetValue(stratStr, out var strat))
                    throw new Exception($"Неизвестна стратегия: {stratStr}");
                if (!GenreMap.TryGetValue(genreStr, out var genre))
                    throw new Exception($"Неизвестен жанр: {genreStr}");

                if (!double.TryParse(sSubjStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var sSubj))
                    throw new Exception($"Невалиден S_subj: {sSubjStr}");
                if (!double.TryParse(sBioStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var sBio))
                    throw new Exception($"Невалиден S_bio: {sBioStr}");
                if (!int.TryParse(nStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n))
                    throw new Exception($"Невалиден n_counts: {nStr}");

                rows.Add(new Row { Mood = mood, Strategy = strat, Genre = genre, SSubj = sSubj, SBio = sBio, NCounts = n });
            }
            return rows;
        }

        private static void EnsureDir(string path)
        {
            if (!Directory.Exists(path)) Directory.CreateDirectory(path);
        }

        // ---------- NOTE изход (подравнени текстови таблици) ----------
        private static void WriteNote(string path, IEnumerable<string[]> rows, string? title = null)
        {
            var list = rows.Select(r => r.Select(c => c ?? "").ToArray()).ToList();
            if (list.Count == 0)
            {
                using var swEmpty = new StreamWriter(path);
                if (!string.IsNullOrEmpty(title)) { swEmpty.WriteLine(title); swEmpty.WriteLine(); }
                swEmpty.WriteLine("(няма данни)");
                return;
            }

            int cols = list.Max(r => r.Length);
            int[] widths = new int[cols];
            foreach (var r in list)
                for (int i = 0; i < cols; i++)
                {
                    string val = i < r.Length ? r[i] : "";
                    widths[i] = Math.Max(widths[i], val.Length);
                }

            using var sw = new StreamWriter(path);
            if (!string.IsNullOrEmpty(title))
            {
                sw.WriteLine(title);
                sw.WriteLine(new string('=', title.Length));
                sw.WriteLine();
            }

            for (int ridx = 0; ridx < list.Count; ridx++)
            {
                var row = list[ridx];
                var cells = new string[cols];
                for (int i = 0; i < cols; i++)
                {
                    string val = i < row.Length ? row[i] : "";
                    cells[i] = val.PadRight(widths[i]);
                }
                sw.WriteLine(" | " + string.Join(" | ", cells) + " |");
                if (ridx == 0)
                {
                    sw.WriteLine(" |-"
                        + string.Join("-|-", widths.Select(w => new string('-', w)))
                        + "-|");
                }
            }
        }

        // ---------- Състояния ----------
        private readonly record struct StateKey(Mood Mood, Strategy? Strategy)
        {
            public override string ToString() =>
                Strategy is null ? MoodBg(Mood) : $"{MoodBg(Mood)} × {StratBg(Strategy.Value)}";
        }

        private static IEnumerable<StateKey> EnumerateStates(Config cfg)
        {
            var moods = Enum.GetValues(typeof(Mood)).Cast<Mood>();
            var strats = Enum.GetValues(typeof(Strategy)).Cast<Strategy>();
            if (cfg.Granularity == Conditioning.Mood)
                foreach (var m in moods) yield return new StateKey(m, null);
            else
                foreach (var m in moods)
                    foreach (var s in strats)
                        yield return new StateKey(m, s);
        }

        private static (double Hcond, List<(StateKey x, double px, double Hx, double Cx)> rows)
            ConditionalEntropySummary(List<Row> rows, Config cfg)
        {
            var states = EnumerateStates(cfg).ToList();

            var Nx = new Dictionary<StateKey, int>();
            foreach (var x in states)
            {
                int n = (cfg.Granularity == Conditioning.Mood)
                    ? rows.Where(r => r.Mood == x.Mood).Sum(r => r.NCounts)
                    : rows.Where(r => r.Mood == x.Mood && r.Strategy == x.Strategy).Sum(r => r.NCounts);
                Nx[x] = n;
            }
            int Ntot = Math.Max(1, Nx.Values.Sum());

            double Hcond = 0.0;
            var outRows = new List<(StateKey, double, double, double)>();
            foreach (var x in states)
            {
                var (u, sMeta, sSubj, sBio, p_gx, Cx, Hx) = UtilitiesFor(rows, cfg, x.Mood, x.Strategy ?? Strategy.Amplify);
                double px = (double)Nx[x] / Ntot;
                Hcond += px * Hx;
                outRows.Add((x, px, Hx, Cx));
            }

            return (Hcond, outRows);
        }

        // ---------- Конзолни помощници ----------
        private static int ReadOption(string prompt, int min, int max)
        {
            while (true)
            {
                Console.Write(prompt);
                var s = Console.ReadLine();
                if (int.TryParse(s, out var v) && v >= min && v <= max) return v;
                Console.WriteLine($"Моля, въведи число между {min} и {max}.");
                Console.WriteLine("");
            }
        }

        private static Mood ReadMood()
        {
            var moods = Enum.GetValues(typeof(Mood)).Cast<Mood>().ToArray();
            Console.WriteLine("Избери настроение:");
            for (int i = 0; i < moods.Length; i++) Console.WriteLine($"  {i + 1}) {MoodBg(moods[i])}");
            int opt = ReadOption("Опция (1-" + moods.Length + "): ", 1, moods.Length);
            Console.WriteLine("");
            return moods[opt - 1];
        }

        private static Strategy ReadStrategy()
        {
            var strats = Enum.GetValues(typeof(Strategy)).Cast<Strategy>().ToArray();
            Console.WriteLine("Избери стратегия:");
            for (int i = 0; i < strats.Length; i++) Console.WriteLine($"  {i + 1}) {StratBg(strats[i])}");
            int opt = ReadOption("Опция (1-" + strats.Length + "): ", 1, strats.Length);
            Console.WriteLine("");
            return strats[opt - 1];
        }

        // ---------- Аргументи ----------
        private static void PrintHelp()
        {
            Console.WriteLine("MusicEvo Console (.NET 8)");
            Console.WriteLine("Аргументи:");
            Console.WriteLine("  --input <path>           Входен CSV (mood,strategy,genre,S_subj,S_bio,n_counts)");
            Console.WriteLine("  --out <dir>              Изходна директория (default: out)");
            Console.WriteLine("  --granularity <g>        mood | mood_strategy (default: mood)");
            Console.WriteLine("  --alpha <val>            Лаплас α (default: 1.0)");
            Console.WriteLine("  --wsubj <val>            w_subj (default: 0.5)");
            Console.WriteLine("  --wbio <val>             w_bio  (default: 0.3)");
            Console.WriteLine("  --wmeta <val>            w_meta (default: 0.2)");
            Console.WriteLine("  --eta <val>              η (default: 0.15)");
            Console.WriteLine("  --tau <val>              τ (default: 0.5)");
            Console.WriteLine("  --mu <val>               μ (default: 1e-3)");
            Console.WriteLine("  --steps <int>            стъпки (default: 10000)");
            Console.WriteLine("  --help                   помощ");
        }

        private static Config ParseArgs(string[] args, Config cfg, ref string input)
        {
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i].ToLowerInvariant();
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"Липсва стойност за {a}");
                switch (a)
                {
                    case "--help": PrintHelp(); Environment.Exit(0); break;
                    case "--input": input = Next(); break;
                    case "--out": cfg = cfg with { OutDir = Next() }; break;
                    case "--granularity":
                        var g = Next();
                        cfg = cfg with
                        {
                            Granularity = g.StartsWith("mood_strategy", StringComparison.OrdinalIgnoreCase)
                                        ? Conditioning.MoodStrategy : Conditioning.Mood
                        };
                        break;
                    case "--alpha": cfg = cfg with { Alpha = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--wsubj": cfg = cfg with { WSubj = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--wbio": cfg = cfg with { WBio = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--wmeta": cfg = cfg with { WMeta = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--eta": cfg = cfg with { Eta = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--tau": cfg = cfg with { Tau = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--mu": cfg = cfg with { Mu = double.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                    case "--steps": cfg = cfg with { Steps = int.Parse(Next(), CultureInfo.InvariantCulture) }; break;
                }
            }
            return cfg;
        }

        // ---------- Main ----------
        private static void Main(string[] args)
        {
            var cfg = new Config();
            string input = @"C:\Users\IvanD\Desktop\realistic_input_music_evo.csv";
            cfg = ParseArgs(args, cfg, ref input);

            Console.WriteLine(" === MusicEvo – C# Реализация на Глава 3 ===");
            Console.WriteLine($"Входен файл: {input}");

            if (!File.Exists(input))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ГРЕШКА] Входният CSV не е намерен или е недостъпен: '{input}'");
                Console.ResetColor();
                Console.WriteLine("Подайте валиден път с --input <path>.");
                return;
            }

            // Четене
            List<Row> data;
            try { data = LoadCsv(input); }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ГРЕШКА] Проблем при четене на CSV: {ex.Message}");
                Console.ResetColor();
                return;
            }

            // Интерактивен избор
            Console.WriteLine();
            Console.WriteLine("Какво искаш да анализираш?");
            Console.WriteLine("  1) Всички състояния");
            Console.WriteLine("  2) Само по настроение");
            Console.WriteLine("  3) Конкретно настроение × стратегия");
            int mode = ReadOption("Опция (1-3): ", 1, 3);

            Mood? chosenMood = null;
            Strategy? chosenStrategy = null;

            Console.WriteLine("");
            if (mode == 1)
            {
                Console.WriteLine("\nИзбери грануларност за p(g|x):");
                Console.WriteLine("  1) По настроение (Mood)");
                Console.WriteLine("  2) По настроение×стратегия (MoodStrategy)");
                int gopt = ReadOption("Опция (1-2): ", 1, 2);
                cfg = cfg with { Granularity = (gopt == 1) ? Conditioning.Mood : Conditioning.MoodStrategy };
            }
            else if (mode == 2)
            {
                chosenMood = ReadMood();
                cfg = cfg with { Granularity = Conditioning.Mood };
            }
            else
            {
                chosenMood = ReadMood();
                chosenStrategy = ReadStrategy();
                cfg = cfg with { Granularity = Conditioning.MoodStrategy };
            }

            Console.WriteLine($"Грануларност p(g|x): {cfg.Granularity}");
            Console.WriteLine($"Параметри: alpha={cfg.Alpha}, w=({cfg.WSubj},{cfg.WBio},{cfg.WMeta}), eta={cfg.Eta}, tau={cfg.Tau}, mu={cfg.Mu}, steps={cfg.Steps}");

            EnsureDir(cfg.OutDir);

            var moods = Enum.GetValues(typeof(Mood)).Cast<Mood>().ToArray();
            var strats = Enum.GetValues(typeof(Strategy)).Cast<Strategy>().ToArray();
            var genres = Enum.GetValues(typeof(Genre)).Cast<Genre>().ToArray();

            // Глобална H(G|X) и локални H~(x)
            var (Hcond, entropyRows) = ConditionalEntropySummary(data, cfg);
            double Cbar = 1.0 - Hcond / Math.Log(K);
            Console.WriteLine($"\nУсловна ентропия H(G|X) = {Hcond:F6}  |  Нормализирана предсказуемост C̄ = {Cbar:F6}");

            var rowsEntropy = new List<string[]>();
            rowsEntropy.Add(new[] { "state", "p(x)", "H_tilde(x)=H(G|X=x)", "C(x)=1-H/lnK" });
            rowsEntropy.AddRange(entropyRows.Select(e => new[]
            {
                e.x.ToString(),
                e.px.ToString("F6", CultureInfo.InvariantCulture),
                e.Hx.ToString("F6", CultureInfo.InvariantCulture),
                e.Cx.ToString("F6", CultureInfo.InvariantCulture),
            }));
            WriteNote(Path.Combine(cfg.OutDir, "entropy_summary.txt"), rowsEntropy, "Entropy Summary");

            // Основни изходи
            var rowsU = new List<string[]>
            {
                new[] { "mood_bg", "strategy_bg" }.Concat(genres.Select(g => $"u[{GenreNameEn(g)}]")).ToArray()
            };
            var rowsX = new List<string[]>
            {
                new[] { "mood_bg", "strategy_bg" }.Concat(genres.Select(g => $"x*[{GenreNameEn(g)}]")).ToArray()
            };
            var rowsTop = new List<string[]>
            {
                new[] { "mood_bg", "strategy_bg", "top1_by_u", "top2_by_u", "top3_by_u", "top1_by_x*", "top2_by_x*", "top3_by_x*" }
            };
            var rowsDetails = new List<string[]>
            {
                new[] { "mood_bg", "strategy_bg", "genre_en", "genre_bg", "p(g|x)", "S_subj", "S_bio", "S_meta", "u" }
            };

            IEnumerable<Mood> loopMoods;
            IEnumerable<Strategy> loopStrats;

            if (mode == 1) { loopMoods = moods; loopStrats = strats; }
            else if (mode == 2) { loopMoods = new[] { chosenMood!.Value }; loopStrats = strats; }
            else { loopMoods = new[] { chosenMood!.Value }; loopStrats = new[] { chosenStrategy!.Value }; }

            foreach (var m in loopMoods)
            {
                foreach (var s in loopStrats)
                {
                    var (u, sMeta, sSubj, sBio, p_gx, Cx, Hx) = UtilitiesFor(data, cfg, m, s);
                    var A = BuildPayoffMatrix(u, cfg.Eta);

                    var xEmp = (double[])p_gx.Clone();
                    var xMod = Softmax(u, cfg.Tau);
                    var x0 = new double[K];
                    for (int i = 0; i < K; i++) x0[i] = (1 - Cx) * xEmp[i] + Cx * xMod[i];

                    var xStar = Replicator(A, cfg, x0);

                    var idxU = Enumerable.Range(0, K).OrderByDescending(i => u[i]).Take(3).ToArray();
                    var idxX = Enumerable.Range(0, K).OrderByDescending(i => xStar[i]).Take(3).ToArray();

                    Console.WriteLine($"\n[{MoodBg(m)} × {StratBg(s)}]  (H~={Hx:F3}, C={Cx:F3})");
                    Console.WriteLine("Топ по u:    " + string.Join(", ", idxU.Select(i => $"{GenreNameEn(genres[i])} ({u[i]:F3})")));
                    Console.WriteLine("Топ по x*:   " + string.Join(", ", idxX.Select(i => $"{GenreNameEn(genres[i])} ({xStar[i]:F3})")));

                    rowsU.Add(new[] { MoodBg(m), StratBg(s) }.Concat(u.Select(v => v.ToString("F6", CultureInfo.InvariantCulture))).ToArray());
                    rowsX.Add(new[] { MoodBg(m), StratBg(s) }.Concat(xStar.Select(v => v.ToString("F6", CultureInfo.InvariantCulture))).ToArray());
                    rowsTop.Add(new[]
                    {
                        MoodBg(m), StratBg(s),
                        GenreNameEn(genres[idxU[0]]), GenreNameEn(genres[idxU[1]]), GenreNameEn(genres[idxU[2]]),
                        GenreNameEn(genres[idxX[0]]), GenreNameEn(genres[idxX[1]]), GenreNameEn(genres[idxX[2]])
                    });

                    for (int g = 0; g < K; g++)
                    {
                        rowsDetails.Add(new[]
                        {
                            MoodBg(m), StratBg(s),
                            GenreNameEn(genres[g]), GenreNameBg(genres[g]),
                            p_gx[g].ToString("F6", CultureInfo.InvariantCulture),
                            sSubj[g].ToString("F6", CultureInfo.InvariantCulture),
                            sBio[g].ToString("F6", CultureInfo.InvariantCulture),
                            sMeta[g].ToString("F6", CultureInfo.InvariantCulture),
                            u[g].ToString("F6", CultureInfo.InvariantCulture)
                        });
                    }
                }
            }

            // Пишем NOTE файлове
            WriteNote(Path.Combine(cfg.OutDir, "utilities_u.txt"), rowsU, "Utilities u[x,g]");
            WriteNote(Path.Combine(cfg.OutDir, "stationary_xstar.txt"), rowsX, "Stationary distribution x*");
            WriteNote(Path.Combine(cfg.OutDir, "top3_by_u_and_xstar.txt"), rowsTop, "Top-3 by u and by x*");
            WriteNote(Path.Combine(cfg.OutDir, "details_by_state.txt"), rowsDetails, "Details by state and genre");

            Console.WriteLine($"\nГотово. Резултатите са записани като NOTE .txt файлове в папка: {cfg.OutDir}");
        }
    }
}
