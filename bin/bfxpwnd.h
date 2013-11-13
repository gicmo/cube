#ifndef BFXP_WINDOW_H
#define BFXP_WINDOW_H

#include <gtk/gtk.h>

G_BEGIN_DECLS

typedef struct _BfXpWindow        BfXpWindow;
typedef struct _BfXpWindowClass   BfXpWindowClass;
typedef struct _BfXpWindowPrivate BfXpWindowPrivate;

#define BFXP_TYPE_WINDOW         (bfxp_window_get_type ())
#define BFXP_WINDOW(i)           (G_TYPE_CHECK_INSTANCE_CAST ((i), BFXP_TYPE_WINDOW, BfXpWindow))
#define BFXP_WINDOW_CLASS(c)     (G_TYPE_CHECK_CLASS_CAST ((c), BFXP_TYPE_WINDOW, BfXpWindowClass))
#define BFXP_IS_WINDOW(i)        (G_TYPE_CHECK_INSTANCE_TYPE ((i), BFXP_TYPE_WINDOW))
#define BFXP_IS_WINDOW_CLASS(c)  (G_TYPE_CHECK_CLASS_TYPE ((c), BFXP_TYPE_WINDOW))
#define BFXP_WINDOW_GET_CLASS(i) (G_TYPE_INSTANCE_GET_CLASS ((i), BFXP_TYPE_WINDOW, BfXpWindowClass))

GType      bfxp_window_get_type     (void);
GtkWidget* bfxp_window_new          (void);
void       bfxp_window_load_data    (BfXpWindow *self);

struct _BfXpWindow
{
  GtkWindow          parent;
  BfXpWindowPrivate *priv;
};

struct _BfXpWindowClass
{
  GtkWindowClass     base;
};

G_END_DECLS

#endif /* !BFXP_WINDOW_H */

/* vim:set et sw=2 cino=t0,f0,(0,{s,>2s,n-1s,^-1s,e2s: */
