package viz

// import (
// 	"fmt"
// 	"os"
// 	"strings"

// 	"github.com/qntx/spark/ag"
// )

// const (
// 	varfmt = "\"%p\" [label=\"%v\", color=orange, style=filled]"
// 	fncfmt = "\"%p\" [label=\"%s\", color=lightblue, style=filled, shape=box]"
// 	arrow  = "\"%p\" -> \"%p\""
// )

// type Opts struct {
// 	Verbose bool
// }

// func Var(v *ag.Var, opts ...Opts) string {
// 	if len(opts) > 0 && opts[0].Verbose {
// 		return fmt.Sprintf(varfmt, v, v)
// 	}

// 	return fmt.Sprintf(varfmt, v, v.Name)
// }

// func Func(f *ag.Operator) []string {
// 	s := f.String()
// 	begin, end := strings.Index(s, "."), strings.Index(s, "T[")
// 	out := []string{fmt.Sprintf(fncfmt, f, s[begin+1:end])}

// 	for _, x := range f.Input {
// 		out = append(out, fmt.Sprintf(arrow, x, f))
// 	}

// 	for _, y := range f.Output {
// 		out = append(out, fmt.Sprintf(arrow, f, y))
// 	}

// 	return out
// }

// func Graph(v *ag.Var, opts ...Opts) []string {
// 	seen := make(map[*ag.Operator]bool)
// 	fs := AddFunc(make([]*ag.Operator, 0), v.Creator, seen)

// 	out := append([]string{"digraph g {"}, Var(v, opts...))
// 	for {
// 		if len(fs) == 0 {
// 			break
// 		}

// 		// pop
// 		f := fs[len(fs)-1]
// 		fs = fs[:len(fs)-1]
// 		out = append(out, Func(f)...)

// 		x := f.Input
// 		for i := range x {
// 			out = append(out, Var(x[i], opts...))

// 			if x[i].Creator != nil {
// 				fs = AddFunc(fs, x[i].Creator, seen)
// 			}
// 		}
// 	}

// 	out = append(out, "}")
// 	return out
// }

// func AddFunc(fs []*ag.Operator, f *ag.Operator, seen map[*ag.Operator]bool) []*ag.Operator {
// 	if _, ok := seen[f]; ok {
// 		return fs
// 	}

// 	seen[f] = true
// 	fs = append(fs, f)
// 	return fs
// }

// // SaveGraph generates a DOT graph for the given variable and saves it to the specified file path
// func SaveGraph(v *ag.Var, filepath string, opts ...Opts) error {
// 	lines := Graph(v, opts...)
// 	content := strings.Join(lines, "\n")

// 	return os.WriteFile(filepath, []byte(content), 0644)
// }
